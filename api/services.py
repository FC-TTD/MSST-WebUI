from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import shutil
import signal
import threading
import time
from typing import Any, Dict, List, Optional

from api.models import FileResult, ModelInfo, TaskCreateRequest, TaskCreateResponse, TaskResultResponse
from api.storage import get_storage
from api.task_results import collect_task_output_files
from api.task_runtime import TaskRuntime, get_cancel_event, get_task_lock, get_task_runtime, register_task_runtime, unregister_task_runtime
from utils.constant import MODELS_INFO, WEBUI_CONFIG


logger = logging.getLogger(__name__)


class InferenceBusyError(RuntimeError):
    pass


inference_semaphore = multiprocessing.Semaphore(2)

_semaphore_count = 0
_semaphore_lock = threading.Lock()

INFERENCE_QUEUE_TIMEOUT_S = float(os.getenv("MSST_QUEUE_TIMEOUT_S", "3600"))


class _QueueCallback:
    def __init__(self, q: multiprocessing.queues.Queue):  # type: ignore[name-defined]
        self._q = q

    def __setitem__(self, key: str, value: Any) -> None:
        try:
            self._q.put((key, value), timeout=1.0)
        except Exception as e:
            # 区分不同类型的异常
            if "full" in str(e).lower():
                logger.warning(f"Queue full, dropping {key} update")
            elif "closed" in str(e).lower() or "broken" in str(e).lower():
                logger.error(f"Queue closed/broken for {key}: {e}")
            else:
                logger.exception(f"Queue error for {key}: {e}")
            # 不再静默忽略，记录错误但继续执行


def _try_acquire_inference_slot() -> bool:
    global _semaphore_count
    try:
        result = bool(inference_semaphore.acquire(block=False))
        if result:
            with _semaphore_lock:
                _semaphore_count += 1
        return result
    except TypeError:
        result = bool(inference_semaphore.acquire(False))
        if result:
            with _semaphore_lock:
                _semaphore_count += 1
        return result


def _release_inference_slot() -> None:
    global _semaphore_count
    with _semaphore_lock:
        if _semaphore_count > 0:
            try:
                inference_semaphore.release()
                _semaphore_count -= 1
            except ValueError:
                # 信号量已经达到最大值，重置计数器
                _semaphore_count = 0


def _terminate_pid(pid: int, *, term_timeout_s: float = 10.0) -> None:
    if pid <= 0:
        return

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        logger.exception(f"Failed to SIGTERM pid={pid}")
        return

    deadline = time.time() + term_timeout_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        except Exception:
            break
        time.sleep(0.05)

    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        logger.exception(f"Failed to SIGKILL pid={pid}")


def _acquire_inference_slot_or_wait(*, cancel_event: threading.Event | None = None, timeout_s: float = INFERENCE_QUEUE_TIMEOUT_S) -> bool:
    global _semaphore_count
    deadline = time.time() + float(timeout_s)
    while True:
        if cancel_event is not None and cancel_event.is_set():
            return False

        remaining = deadline - time.time()
        if remaining <= 0:
            return False

        step = 1.0 if remaining > 1.0 else remaining
        try:
            ok = bool(inference_semaphore.acquire(timeout=step))
        except TypeError:
            ok = bool(inference_semaphore.acquire(True, step))

        if ok:
            # 维护信号量计数器，防止泄漏
            with _semaphore_lock:
                _semaphore_count += 1
            return True


def _i18n(msg: str) -> str:
    try:
        from webui.utils import i18n as _real_i18n

        return _real_i18n(msg)
    except Exception:
        return msg


def run_folder_batch_inference(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError("webui dependencies are not available")


def run_inference(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError("webui dependencies are not available")


def _validate_host_path(path: str) -> str | None:
    if path.startswith("/Volume/") or path.startswith("/Volumes/"):
        return _i18n("非法路径（疑似 macOS Volume 路径）: ") + path
    if re.match(r"^[A-Za-z]:[\\/]", path):
        return _i18n("非法路径（疑似 Windows 盘符路径）: ") + path
    return None


def _finalize_task_output(task_output_dir: str, output_dir: str) -> None:
    if not task_output_dir or not os.path.exists(task_output_dir):
        return

    os.makedirs(output_dir, exist_ok=True)

    for root, dirnames, filenames in os.walk(task_output_dir):
        dirnames[:] = [d for d in dirnames if not d.startswith(("@", "."))]
        filenames = [f for f in filenames if not f.startswith(("@", "."))]
        for filename in filenames:
            src = os.path.join(root, filename)
            rel_path = os.path.relpath(src, task_output_dir)
            dst = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                os.replace(src, dst)
            except Exception:
                try:
                    shutil.copy2(src, dst)
                    os.remove(src)
                except Exception:
                    logger.exception(f"Failed to move output file: {src} -> {dst}")

    shutil.rmtree(task_output_dir, ignore_errors=True)


def cancel_msst_sse_task(task_id: str) -> TaskResultResponse | None:
    runtime = get_task_runtime(task_id)
    if not runtime:
        storage = get_storage()
        status = storage.get_task_status(task_id)
        if not status:
            return None

        cancel_event = get_cancel_event(task_id)
        cancel_event.set()
        storage.update_task(task_id, status="canceled")
        return TaskResultResponse(task_id=task_id, status="canceled", files=[])

    lock = get_task_lock(task_id)
    with lock:
        runtime = get_task_runtime(task_id)
        if not runtime:
            return None

        storage = get_storage()

        cancel_event = get_cancel_event(task_id)
        cancel_event.set()

        try:
            _terminate_pid(runtime.pid)
        except ProcessLookupError:
            pass
        except Exception:
            logger.exception(f"Failed to terminate task process: task_id={task_id}, pid={runtime.pid}")

        files: List[FileResult] = []
        try:
            files = collect_task_output_files(runtime.task_output_dir)
            storage.save_results(task_id, files)
        except Exception:
            logger.exception(f"Failed to collect/save partial results for canceled task: task_id={task_id}")

        try:
            _finalize_task_output(runtime.task_output_dir, runtime.output_dir)
        except Exception:
            logger.exception(f"Failed to finalize output dir for canceled task: task_id={task_id}")

        storage.update_task(task_id, status="canceled")

        unregister_task_runtime(task_id)

        return TaskResultResponse(task_id=task_id, status="canceled", files=files)


def run_msst_batch_sync(req: TaskCreateRequest) -> TaskResultResponse:
    """同步（blocking）执行一次批量 MSST 分离，并记录任务结果。

    说明：
    - 当前实现直接复用 webui.msst.run_folder_batch_inference，保持与 Gradio 相同的推理链路；
    - API 允许调用方自定义 input_path/output_dir，但要求服务端能访问这些路径；
    - 为简化首版实现，暂不返回过程进度，仅在完成后一次性返回结果。
    """

    storage = get_storage()

    # 先创建任务记录，初始状态为 running
    task_id = storage.create_task(status="running", message=None)

    input_path = os.path.abspath(req.input_path)
    output_dir = os.path.abspath(req.output_dir)

    path_err = _validate_host_path(req.input_path) or _validate_host_path(req.output_dir)
    if path_err:
        logger.error(path_err)
        storage.update_task(task_id, status="failed", error=path_err)
        return TaskResultResponse(task_id=task_id, status="failed", files=[])

    # 简单校验路径
    if not os.path.exists(input_path):
        msg = _i18n("输入路径不存在: ") + input_path
        logger.error(msg)
        storage.update_task(task_id, status="failed", error=msg)
        return TaskResultResponse(task_id=task_id, status="failed", files=[])

    os.makedirs(output_dir, exist_ok=True)

    range_start_raw = req.params.get("range_start")
    range_end_raw = req.params.get("range_end")
    input_path_for_infer = input_path
    cleanup_dir: str | None = None

    if os.path.isfile(input_path):
        import tempfile
        import uuid

        # 使用uuid确保临时目录名唯一，避免并发冲突
        unique_id = str(uuid.uuid4())[:8]
        tmp_base = os.path.join(tempfile.gettempdir(), f".msst_api_single_{task_id}_{unique_id}")
        os.makedirs(tmp_base, exist_ok=True)
        src = input_path
        dst = os.path.join(tmp_base, os.path.basename(src))
        try:
            os.symlink(src, dst)
        except Exception:
            shutil.copy2(src, dst)
        input_path_for_infer = tmp_base
        cleanup_dir = tmp_base

    if os.path.isdir(input_path) and (range_start_raw is not None or range_end_raw is not None):
        try:
            range_start = int(range_start_raw) if range_start_raw is not None else 1
            range_end = int(range_end_raw) if range_end_raw is not None else 0
        except Exception:
            msg = _i18n("range_start/range_end 必须为整数")
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            return TaskResultResponse(task_id=task_id, status="failed", files=[])

        if range_start < 1:
            range_start = 1

        names = sorted(os.listdir(input_path))
        # 过滤出音频/视频文件（文件场景）或保留子目录（项目/剧集场景），避免非媒体文件和NAS系统文件干扰 Range 索引
        filtered_names = []
        for name in names:
            # 跳过以@或.开头的文件（NAS系统文件）
            if name.startswith(('@', '.')):
                continue
            file_path = os.path.join(input_path, name)
            if os.path.isdir(file_path):
                filtered_names.append(name)
                continue
            if os.path.isfile(file_path):
                # 检查文件扩展名是否为音频或视频格式
                # librosa支持通过ffmpeg提取视频中的音频
                if name.lower().endswith(
                    (
                        '.wav',
                        '.mp3',
                        '.flac',
                        '.m4a',
                        '.aac',
                        '.ogg',
                        '.mp4',
                        '.avi',
                        '.mkv',
                        '.mov',
                        '.wmv',
                        '.flv',
                        '.webm',
                    )
                ):
                    filtered_names.append(name)
        names = filtered_names
        total = len(names)
        if total == 0:
            msg = _i18n("输入目录中没有找到媒体文件: ") + input_path
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            return TaskResultResponse(task_id=task_id, status="failed", files=[])

        if range_end_raw is None:
            range_end = total

        if range_end < range_start or range_start > total:
            msg = _i18n("range_start/range_end 范围无效")
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            return TaskResultResponse(task_id=task_id, status="failed", files=[])

        if range_end > total:
            range_end = total

        if not (range_start == 1 and range_end == total):
            import tempfile
            tmp_base = os.path.join(tempfile.gettempdir(), f".msst_api_range_{task_id}")
            os.makedirs(tmp_base, exist_ok=True)
            for idx in range(range_start - 1, range_end):
                name = names[idx]
                src = os.path.join(input_path, name)
                dst = os.path.join(tmp_base, name)
                try:
                    if os.path.isdir(src):
                        os.symlink(src, dst, target_is_directory=True)
                    else:
                        os.symlink(src, dst)
                except Exception:
                    if os.path.isdir(src):
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)
            input_path_for_infer = tmp_base
            cleanup_dir = tmp_base

    # 从顶级字段和 params 中获取可选参数，提供合理默认值
    # 输出音轨优先使用顶级字段，兼容旧版 params["instrumental"] 写法
    extract_instrumental: List[str] = (req.extract_instrumental or req.params.get("instrumental") or [])

    # 设备选择：默认不使用 CPU，缺省时自动使用 GPU0
    gpu_id = req.params.get("device_ids")
    force_cpu = bool(req.params.get("force_cpu", False))
    if not force_cpu and (not gpu_id):
        gpu_id = [0]

    output_format = req.params.get("output_format") or "wav"
    # 复用 WebUI 已保存的 use_tta 设置，忽略 API 传参
    try:
        from webui.utils import load_configs

        _webui_cfg = load_configs(WEBUI_CONFIG)
        use_tta = bool(_webui_cfg.get("inference", {}).get("use_tta", True))
    except Exception:
        use_tta = True

    task_output_dir = os.path.join(output_dir, f"task_{task_id}")

    try:
        os.makedirs(task_output_dir, exist_ok=True)

        message, _ = run_folder_batch_inference(
            req.model_name,
            input_path_for_infer,
            task_output_dir,
            extract_instrumental,
            gpu_id,
            output_format,
            force_cpu,
            use_tta,
        )

        logger.info(message)
        storage.update_task(task_id, status="success", progress=1.0)

        # 收集实际生成的输出文件
        files: List[FileResult] = []
        try:
            # 扫描当前任务的输出目录，收集新生成的文件
            if os.path.exists(task_output_dir):
                output_files = []
                for root, dirnames, filenames in os.walk(task_output_dir):
                    dirnames[:] = [d for d in dirnames if not d.startswith(("@", "."))]
                    filenames = [f for f in filenames if not f.startswith(("@", "."))]
                    for filename in filenames:
                        if filename.endswith(('.wav', '.mp3')):
                            file_path = os.path.join(root, filename)
                            output_files.append(os.path.basename(file_path))
                
                # 根据文件名推断输入文件并分组
                input_file_groups = {}
                for output_file in output_files:
                    # 尝试从输出文件名推断输入文件
                    # 例如: test1_Vocals.wav -> test1
                    input_name = None
                    if "_Vocals" in output_file:
                        input_name = output_file.split("_Vocals")[0]
                    elif "_Instrumental" in output_file:
                        input_name = output_file.split("_Instrumental")[0]
                    else:
                        # 如果无法推断，使用通用名称
                        input_name = f"file_{len(input_file_groups) + 1}"
                    
                    if input_name not in input_file_groups:
                        input_file_groups[input_name] = []
                    input_file_groups[input_name].append(output_file)
                
                # 为每个输入文件创建一个 FileResult
                for input_name, files_list in input_file_groups.items():
                    files.append(FileResult(
                        input_file=input_name,
                        output_files=files_list,
                        status="success"
                    ))
                    
            logger.info(f"Collected {len(files)} file results from {task_output_dir}")
        except Exception as e:
            logger.warning(f"Failed to collect output files: {e}")
        
        storage.save_results(task_id, files)
        return TaskResultResponse(task_id=task_id, status="success", files=files)
    except Exception as e:
        err_msg = f"MSST batch inference failed: {e}"
        logger.exception(err_msg)
        storage.update_task(task_id, status="failed", error=err_msg)
        return TaskResultResponse(task_id=task_id, status="failed", files=[])
    finally:
        _finalize_task_output(task_output_dir, output_dir)
        if cleanup_dir:
            shutil.rmtree(cleanup_dir, ignore_errors=True)
        _release_inference_slot()


def list_models(model_class: str | None = None) -> List[ModelInfo]:
    """列出可用模型。

    - 基于 MODELS_INFO（data/models_info.json）
    - 可选按 model_class 过滤，例如 "VR_Models"、"multi_stem_models" 等。
    """

    try:
        from webui.utils import load_configs as _load_configs  # 避免和上面 import 冲突
    except Exception as e:
        raise RuntimeError("webui dependencies are not available") from e

    try:
        config = _load_configs(MODELS_INFO)
    except Exception as e:  # pragma: no cover - 防御性分支
        logger.exception("Failed to load models info")
        raise

    results: List[ModelInfo] = []
    for key, info in config.items():
        mc = info.get("model_class")
        if model_class and mc != model_class:
            continue
        base_fields = {
            "id": key,
            "model_name": info.get("model_name", key),
            "model_class": mc or "",
            "model_size": info.get("model_size"),
            "sha256": info.get("sha256"),
            "is_installed": info.get("is_installed"),
            "target_position": info.get("target_position"),
            "link": info.get("link"),
        }
        extra = {k: v for k, v in info.items() if k not in base_fields}
        results.append(ModelInfo(extra=extra, **base_fields))
    return results


def _build_store_dict(store_dir: str, extract_instrumental: List[str], config_path: str) -> Dict[str, str]:
    """根据选中的音轨构建 store_dict，逻辑参考 webui.msst.start_inference。

    如果未选择任何有效音轨，则默认导出模型支持的全部音轨到同一目录。
    """

    from webui.utils import load_configs as _load_configs  # 避免循环引用

    if isinstance(store_dir, str):
        store_dict: Dict[str, str] = {}
        model_config = _load_configs(config_path)
        for inst in extract_instrumental:
            if inst in model_config.training.get("instruments"):
                store_dict[inst] = store_dir
        if not store_dict:
            store_dict = {k: store_dir for k in model_config.training.get("instruments")}
        return store_dict
    return store_dir  # 兼容未来扩展


def _sse_event(event: str, data: Dict) -> str:
    """构造标准 SSE 事件字符串。"""

    return f"event: {event}\n" f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def run_msst_batch_sse(req: TaskCreateRequest, existing_task_id: str | None = None):
    """以 SSE 事件流的方式执行一次批量 MSST 分离。

    - 使用 multiprocessing + webui.msst.run_inference 复用现有推理逻辑；
    - 通过 Manager().dict() 回调读取进度，更新 SQLite，并推送 SSE 事件；
    - 仍然创建 Task 记录，便于后续通过 /tasks/{task_id} 查询。
    """

    storage = get_storage()
    task_id = existing_task_id or storage.create_task(status="queued", message=None)

    def _gen():
        proc: multiprocessing.Process | None = None
        output_dir: str = ""
        task_output_dir: str = ""
        cleanup_dir: str | None = None
        acquired: bool = False

        try:
            from webui.msst import run_inference as _run_inference
            from webui.utils import get_msst_model, load_configs
        except Exception as e:
            raise RuntimeError("webui dependencies are not available") from e

        input_path = os.path.abspath(req.input_path)
        output_dir = os.path.abspath(req.output_dir)

        path_err = _validate_host_path(req.input_path) or _validate_host_path(req.output_dir)
        if path_err:
            logger.error(path_err)
            storage.update_task(task_id, status="failed", error=path_err)
            yield _sse_event("error", {"task_id": task_id, "message": path_err})
            return

        if not os.path.exists(input_path):
            msg = _i18n("输入路径不存在: ") + input_path
            logger.error(msg)
            storage.update_task(task_id, status="failed", error=msg)
            yield _sse_event("error", {"task_id": task_id, "message": msg})
            return

        cancel_event = get_cancel_event(task_id)

        yield _sse_event("queued", {"task_id": task_id, "status": "queued"})

        if not _acquire_inference_slot_or_wait(cancel_event=cancel_event, timeout_s=INFERENCE_QUEUE_TIMEOUT_S):
            if cancel_event.is_set():
                storage.update_task(task_id, status="canceled")
                yield _sse_event("canceled", {"task_id": task_id, "status": "canceled", "files": []})
                return
            raise InferenceBusyError("inference capacity reached")

        acquired = True
        storage.update_task(task_id, status="running")

        run_inference = _run_inference

        os.makedirs(output_dir, exist_ok=True)
        task_output_dir = os.path.join(output_dir, f"task_{task_id}")

        range_start_raw = req.params.get("range_start")
        range_end_raw = req.params.get("range_end")
        input_path_for_infer = input_path

        if os.path.isfile(input_path):
            import tempfile

            tmp_base = os.path.join(tempfile.gettempdir(), f".msst_api_single_{task_id}")
            os.makedirs(tmp_base, exist_ok=True)
            src = input_path
            dst = os.path.join(tmp_base, os.path.basename(src))
            try:
                os.symlink(src, dst)
            except Exception:
                shutil.copy2(src, dst)
            input_path_for_infer = tmp_base
            cleanup_dir = tmp_base

        if os.path.isdir(input_path) and (range_start_raw is not None or range_end_raw is not None):
            try:
                range_start = int(range_start_raw) if range_start_raw is not None else 1
                range_end = int(range_end_raw) if range_end_raw is not None else 0
            except Exception:
                msg = _i18n("range_start/range_end 必须为整数")
                logger.error(msg)
                storage.update_task(task_id, status="failed", error=msg)
                yield _sse_event("error", {"task_id": task_id, "message": msg})
                return

            if range_start < 1:
                range_start = 1

            names = sorted(os.listdir(input_path))
            filtered_names = []
            for name in names:
                if name.startswith(('@', '.')):
                    continue
                file_path = os.path.join(input_path, name)
                if os.path.isdir(file_path):
                    filtered_names.append(name)
                    continue
                if os.path.isfile(file_path):
                    if name.lower().endswith(
                        (
                            '.wav',
                            '.mp3',
                            '.flac',
                            '.m4a',
                            '.aac',
                            '.ogg',
                            '.mp4',
                            '.avi',
                            '.mkv',
                            '.mov',
                            '.wmv',
                            '.flv',
                            '.webm',
                        )
                    ):
                        filtered_names.append(name)
            names = filtered_names
            total = len(names)
            if total == 0:
                msg = _i18n("输入目录中没有找到音频文件: ") + input_path
                logger.error(msg)
                storage.update_task(task_id, status="failed", error=msg)
                yield _sse_event("error", {"task_id": task_id, "message": msg})
                return

            if range_end_raw is None:
                range_end = total

            if range_end < range_start or range_start > total:
                msg = _i18n("range_start/range_end 范围无效")
                logger.error(msg)
                storage.update_task(task_id, status="failed", error=msg)
                yield _sse_event("error", {"task_id": task_id, "message": msg})
                return

            if range_end > total:
                range_end = total

            if not (range_start == 1 and range_end == total):
                import tempfile

                tmp_base = os.path.join(tempfile.gettempdir(), f".msst_api_range_{task_id}")
                os.makedirs(tmp_base, exist_ok=True)
                for idx in range(range_start - 1, range_end):
                    name = names[idx]
                    src = os.path.join(input_path, name)
                    dst = os.path.join(tmp_base, name)
                    try:
                        if os.path.isdir(src):
                            os.symlink(src, dst, target_is_directory=True)
                        else:
                            os.symlink(src, dst)
                    except Exception:
                        if os.path.isdir(src):
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)
                input_path_for_infer = tmp_base
                cleanup_dir = tmp_base

        extract_instrumental: List[str] = (req.extract_instrumental or req.params.get("instrumental") or [])
        gpu_id = req.params.get("device_ids")
        output_format = req.params.get("output_format") or "wav"
        force_cpu = bool(req.params.get("force_cpu", False))
        try:
            _webui_cfg = load_configs(WEBUI_CONFIG)
            use_tta = bool(_webui_cfg.get("inference", {}).get("use_tta", True))
        except Exception:
            use_tta = True

        try:
            if not req.model_name:
                msg = _i18n("请选择模型")
                storage.update_task(task_id, status="failed", error=msg)
                yield _sse_event("error", {"task_id": task_id, "message": msg})
                return

            gpu_ids: List[int] = []
            if not force_cpu:
                if not gpu_id:
                    gpu_id = [0]
                try:
                    for gpu in gpu_id:
                        gpu_ids.append(int(str(gpu).split(":", 1)[0].replace("cuda", "")))
                except Exception:
                    gpu_ids = [0]
            else:
                gpu_ids = [0]

            gpu_ids = list(set(gpu_ids))
            device = "auto" if not force_cpu else "cpu"

            model_path, config_path, model_type, _ = get_msst_model(req.model_name)
            webui_config = load_configs(WEBUI_CONFIG)
            debug = webui_config["settings"].get("debug", False)
            wav_bit_depth = webui_config["settings"].get("wav_bit_depth", "FLOAT")
            flac_bit_depth = webui_config["settings"].get("flac_bit_depth", "PCM_24")
            mp3_bit_rate = webui_config["settings"].get("mp3_bit_rate", "320k")

            os.makedirs(task_output_dir, exist_ok=True)
            store_dict = _build_store_dict(task_output_dir, extract_instrumental, config_path)

            start_time = time.time()
            logger.info("Starting MSST inference process (SSE mode)...")

            yield _sse_event("start", {"task_id": task_id, "status": "running"})

            q: multiprocessing.Queue = multiprocessing.Queue()
            callback = _QueueCallback(q)
            info: Dict[str, Any] = {"index": -1, "total": -1, "name": ""}
            progress: float = 0.0
            flag: Any = (0, None)

            proc = multiprocessing.Process(
                target=run_inference,
                args=(
                    model_type,
                    config_path,
                    model_path,
                    device,
                    gpu_ids,
                    output_format,
                    use_tta,
                    store_dict,
                    debug,
                    wav_bit_depth,
                    flac_bit_depth,
                    mp3_bit_rate,
                    input_path_for_infer,
                    callback,
                    "recursive" if req.params.get("batch_mode") == "recursive" else "folder",
                ),
                name="msst_inference_api",
            )

            proc.start()
            logger.debug(f"Inference process (SSE) started, PID: {proc.pid}")

            register_task_runtime(
                TaskRuntime(
                    task_id=task_id,
                    pid=int(proc.pid or 0),
                    task_output_dir=task_output_dir,
                    output_dir=output_dir,
                )
            )

            def _drain_queue() -> None:
                nonlocal info, progress, flag
                while True:
                    try:
                        k, v = q.get_nowait()
                    except Exception:
                        break
                    if k == "info":
                        try:
                            info = dict(v)
                        except Exception:
                            pass
                    elif k == "progress":
                        try:
                            progress = float(v)
                        except Exception:
                            pass
                    elif k == "flag":
                        flag = v

            while proc.is_alive() and progress < 1.0:
                _drain_queue()

                if cancel_event.is_set():
                    try:
                        if proc.pid:
                            _terminate_pid(int(proc.pid))
                    except ProcessLookupError:
                        pass
                    except Exception:
                        logger.exception(f"Failed to terminate SSE subprocess: task_id={task_id}, pid={proc.pid}")
                    break

                if isinstance(flag, tuple) and flag and flag[0]:
                    break

                if info.get("index", -1) != -1:
                    try:
                        processed = int(info.get("index") or 0)
                    except (ValueError, TypeError):
                        processed = 0
                        logger.warning(f"Invalid index value: {info.get('index')}")
                    
                    try:
                        total = int(info.get("total") or 0)
                    except (ValueError, TypeError):
                        total = 0
                        logger.warning(f"Invalid total value: {info.get('total')}")
                    
                    current_file = str(info.get("name") or "")
                    
                    # 验证数据合理性
                    if processed < 0:
                        processed = 0
                    if total < 0:
                        total = 0
                    
                    storage.update_task(
                        task_id,
                        progress=progress,
                        total_files=total,
                        processed_files=processed,
                    )
                    yield _sse_event(
                        "progress",
                        {
                            "task_id": task_id,
                            "status": "running",
                            "progress": progress,
                            "processed_files": processed,
                            "total_files": total,
                            "current_file": current_file,
                        },
                    )

                time.sleep(0.1)

            if cancel_event.is_set():
                proc.join(timeout=3)
                if proc.is_alive() and proc.pid:
                    _terminate_pid(int(proc.pid))
                    proc.join(timeout=3)
            else:
                proc.join()

            _drain_queue()

            if cancel_event.is_set():
                stored = storage.get_task_result(task_id)
                result = stored
                if result is None or result.status != "canceled":
                    result = cancel_msst_sse_task(task_id)
                if result is None:
                    storage.update_task(task_id, status="canceled")
                    yield _sse_event("canceled", {"task_id": task_id, "status": "canceled", "files": []})
                    return

                files_payload = [
                    (f.model_dump() if hasattr(f, "model_dump") else f.dict())  # type: ignore[attr-defined]
                    for f in result.files
                ]
                yield _sse_event(
                    "canceled",
                    {"task_id": task_id, "status": "canceled", "files": files_payload},
                )
                return

            if flag[0] == 1:
                duration = round(time.time() - start_time, 2)
                storage.update_task(task_id, status="success", progress=1.0)

                files: List[FileResult] = []
                try:
                    files = collect_task_output_files(task_output_dir)
                    logger.info(f"Collected {len(files)} file results from {task_output_dir}")
                    storage.save_results(task_id, files)
                except Exception as e:
                    logger.warning(f"Failed to collect output files: {e}")

                _finalize_task_output(task_output_dir, output_dir)

                files_payload = [
                    (f.model_dump() if hasattr(f, "model_dump") else f.dict())  # type: ignore[attr-defined]
                    for f in files
                ]
                yield _sse_event(
                    "completed",
                    {"task_id": task_id, "status": "success", "duration": duration, "files": files_payload},
                )
                unregister_task_runtime(task_id)
            elif flag[0] == -1:
                err_msg = str(flag[1])
                storage.update_task(task_id, status="failed", error=err_msg)
                _finalize_task_output(task_output_dir, output_dir)
                yield _sse_event(
                    "error",
                    {"task_id": task_id, "status": "failed", "message": err_msg},
                )
                unregister_task_runtime(task_id)
            else:
                msg = "MSST inference process exited unexpectedly"
                storage.update_task(task_id, status="failed", error=msg)
                _finalize_task_output(task_output_dir, output_dir)
                yield _sse_event(
                    "error",
                    {"task_id": task_id, "status": "failed", "message": msg},
                )
                unregister_task_runtime(task_id)
        except Exception as e:  # pragma: no cover
            err_msg = f"MSST batch inference (SSE) failed: {e}"
            logger.exception(err_msg)
            storage.update_task(task_id, status="failed", error=err_msg)
            try:
                _finalize_task_output(task_output_dir, output_dir)
            except Exception:
                logger.exception(f"Failed to finalize output dir after SSE error: task_id={task_id}")
            yield _sse_event(
                "error",
                {"task_id": task_id, "status": "failed", "message": err_msg},
            )
        finally:
            try:
                if proc is not None and proc.is_alive():
                    try:
                        if proc.pid:
                            _terminate_pid(int(proc.pid))
                    finally:
                        proc.join(timeout=3)
            except Exception:
                logger.exception(f"Failed during SSE finalization: task_id={task_id}")

            unregister_task_runtime(task_id)
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

            if task_output_dir and output_dir:
                try:
                    _finalize_task_output(task_output_dir, output_dir)
                except Exception:
                    logger.exception(f"Failed to finalize output dir in SSE finally: task_id={task_id}")

            if acquired:
                _release_inference_slot()

    return _gen()


def start_msst_batch_async(req: TaskCreateRequest) -> TaskCreateResponse:
    """Start the existing SSE lifecycle in a background thread and return its stable task ID."""

    storage = get_storage()
    task_id = storage.create_task(status="queued", message=None)

    def _run() -> None:
        try:
            for _event in run_msst_batch_sse(req, existing_task_id=task_id):
                pass
        except Exception as error:  # pragma: no cover - last-resort task state guard
            logger.exception("MSST async task failed before lifecycle finalization: task_id=%s", task_id)
            storage.update_task(task_id, status="failed", error=str(error))

    threading.Thread(target=_run, name=f"msst_async_{task_id}", daemon=True).start()
    return TaskCreateResponse(task_id=task_id, status="queued", message=None)
