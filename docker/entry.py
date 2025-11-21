import json
import logging
import multiprocessing
import os
import shutil

from fastapi import FastAPI
import gradio as gr
from torch import backends, cuda
from ttd_fastapi_utils import setup_cuda_health
import uvicorn

from api.app import register_routes
from utils.constant import PACKAGE_VERSION, THEME_FOLDER
from webui.utils import i18n, logger
multiprocessing.set_start_method("spawn", force=True)


def build_devices():
    devices = {}
    force_cpu = False
    if cuda.is_available():
        for i in range(cuda.device_count()):
            devices[f"cuda{i}"] = f"{i}: {cuda.get_device_name(i)}"
        logger.info(i18n("检测到CUDA, 设备信息: ") + str(devices))
    elif backends.mps.is_available():
        devices = {"mps": i18n("使用MPS")}
        logger.info(i18n("检测到MPS, 使用MPS"))
    else:
        devices = {"cpu": i18n("无可用的加速设备, 使用CPU")}
        logger.warning(i18n("\033[33m未检测到可用的加速设备, 使用CPU\033[0m"))
        logger.warning(i18n("\033[33m如果你使用的是NVIDIA显卡, 请更新显卡驱动至最新版后重试\033[0m"))
        force_cpu = True
    return devices, force_cpu


def create_app():

    from webui import app as webui_app_module
    from webui.setup import setup_webui

    # 初始化环境与配置
    webui_config = setup_webui()

    # 设备信息
    devices, force_cpu = build_devices()

    # 主题路径
    theme_path = os.path.join(THEME_FOLDER, webui_config["settings"].get("theme", "theme_blue.json"))

    # 构建 Gradio Blocks
    demo = webui_app_module.app(platform=f"Docker, PACKAGE {PACKAGE_VERSION}", device=devices, force_cpu=force_cpu, theme=theme_path).queue()

    # FastAPI 应用和健康检查
    fastapi_app = FastAPI()

    # 添加 CUDA 健康检查
    setup_cuda_health(fastapi_app, ready_predicate=lambda: cuda.is_available())

    # 注册 API 路由（必须在 Gradio 挂载之前）
    register_routes(fastapi_app)

    # 将 Gradio 挂载到根路径
    gr.mount_gradio_app(fastapi_app, demo, path="/")

    return fastapi_app


if __name__ == "__main__":
    
    logger.info(i18n("正在启动容器内的 WebUI 服务 (FastAPI+Uvicorn)..."))
    host = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    try:
        port = int(os.environ.get("PORT", "7860"))
    except ValueError:
        port = 7860

    if not os.path.exists("configs"):
        shutil.copytree("configs_backup", "configs")
        logger.info(i18n("配置文件已复制"))
    else:
        logger.info(i18n("配置文件已存在, 跳过复制"))
    if not os.path.exists("data"):
        shutil.copytree("data_backup", "data")
        logger.info(i18n("数据文件已复制"))
    else:
        logger.info(i18n("数据文件已存在, 跳过复制"))

    app = create_app()

    uvicorn.run(app, host=host, port=port, log_level="info")
