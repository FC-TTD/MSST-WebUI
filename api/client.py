"""Python client for MSST-WebUI API based on `requests`.

封装了：
- 同步批量分离（mode=sync）
- SSE 流式批量分离（mode=sse）
- 任务状态与结果查询
- 模型列表查询（含按 `model_class` 过滤）

仅依赖 `requests`，适合嵌入到任意 Python 项目或 AI agent 中。

默认不设置网络超时（``timeout=None``），便于处理耗时较长的分离任务；
如有网关/代理要求，可在初始化时显式传入秒级超时。 

快速使用示例（同步调用）::

    from api.client import MSSTApiClient

    client = MSSTApiClient()  # 默认指向 docker-compose 暴露的 http://ttd-edge:8662/

    # 推荐：通用人声分离，台词优秀，ME 尚可
    result = client.create_batch_sync(
        input_path="/TTD/04_应用/msst/INPUT",
        output_dir="/TTD/04_应用/msst/OUTPUT",
        model_type="vocal_models",
        model_name="melband_roformer_instvox_duality_v2.ckpt",
        extract_instrumental=["Vocals", "Instrumental"],
        range_start=1,
        range_end=6,
    )
    print(result["status"], result["task_id"])

SSE 流式调用示例（适合长任务与前端/agent 实时感知进度）::

    from api.client import MSSTApiClient

    client = MSSTApiClient()
    for event in client.create_batch_sse(
        input_path="/TTD/04_应用/msst/INPUT_BATCH",
        output_dir="/TTD/04_应用/msst/OUTPUT",
        model_type="vocal_models",
        # 推荐：加强 ME/伴奏质量，国际声场景优秀
        model_name="mel_band_roformer_instrumental_becruily.ckpt",
        extract_instrumental=["Vocals", "Instrumental"],
        range_start=1,
        range_end=6,
    ):
        # event.event 可能是 "start" / "progress" / "completed" / "error" / None
        # 当调用 cancel 接口中断任务时，服务端会发送 "canceled" 事件，并在 event.data.files 返回已完成的部分结果
        print(event.event, event.data)

项目级处理示例（支持任意项目目录结构）::

    from api.client import MSSTApiClient

    client = MSSTApiClient()
    
    # 处理项目目录下的剧集文件
    resp = client.create_batch_sync(
        input_path="/TTD/00_项目/demo_project/input",           # 项目输入目录
        output_dir="/TTD/00_项目/demo_project/03_Stem/demo_project",  # 项目输出目录
        model_type="vocal_models",
        model_name="melband_roformer_instvox_duality_v2.ckpt",
        extract_instrumental=["Vocals", "Instrumental"],
        params={"device_ids": ["cuda:0"], "use_tta": False},
    )
    
    # 获取处理结果
    task_id = resp["task_id"]
    result = client.get_task_result(task_id)
    print(f"任务状态: {result['status']}")
    
    for file_result in result["files"]:
        input_file = file_result["input_file"]
        output_files = file_result["output_files"]
        print(f"输入文件: {input_file}")
        for output_file in output_files:
            print(f"  输出: {output_file}")

模型列表与模型发现示例::

    # 列出 vocal_models 分类下的所有模型
    models = client.list_models(model_class="vocal_models")
    for m in models["models"]:
        print(m["id"], m["model_name"], m["extra"].get("primary_stem"))

AI agent 使用建议：

- 首选通过 :meth:`MSSTApiClient.list_models` 发现可用模型，再根据 `model_class` / `extra` 字段做自动选择。
- 如无特殊要求，可默认使用：
  - ``vocal_models / melband_roformer_instvox_duality_v2.ckpt`` 作为**通用台词人声模型**；
  - ``vocal_models / mel_band_roformer_instrumental_becruily.ckpt`` 作为**加强 ME/伴奏质量**的模型。
- 调用批量分离时，优先使用绝对路径，确保与容器卷挂载配置一致。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterable, Optional

import requests


@dataclass
class SSEEvent:
    """单条 SSE 事件。"""

    event: Optional[str]
    data: Dict[str, Any]


class MSSTApiError(Exception):
    """MSST API 调用异常。"""

    def __init__(self, message: str, status_code: int | None = None, body: Any | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class MSSTApiClient:
    """MSST-WebUI API 的简单 Python 客户端。

    典型调用流程：

    1. 初始化 client（可覆盖 ``base_url`` 和 ``timeout``::

           # 默认 timeout=None 表示不限制请求时间
           client = MSSTApiClient(base_url="http://ttd-edge:8662/", timeout=None)

    2. 选择调用模式：

       - :meth:`create_batch_sync` 适合脚本/离线批量处理；
       - :meth:`create_batch_sse` 适合需要实时进度的前端或 agent；
       - :meth:`get_task_status` / :meth:`get_task_result` 用于轮询。

    3. 模型选择建议：

       - 默认通用人声：``model_type="vocal_models"`` 且
         ``model_name="melband_roformer_instvox_duality_v2.ckpt"``；
       - 加强 ME/伴奏质量：``model_type="vocal_models"`` 且
         ``model_name="mel_band_roformer_instrumental_becruily.ckpt"``。

    该类在异常时抛出 :class:`MSSTApiError`，便于上层统一捕获和重试控制。
    """

    def __init__(
        self,
        base_url: str = "http://ttd-edge:8662/",
        timeout: Optional[float] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        # 去掉末尾斜杠，避免重复 //
        self.base_url = base_url.rstrip("/")
        # timeout=None 表示不限制请求时间（适合长时间分离任务）
        self.timeout = timeout
        self._session = session or requests.Session()

    # --------
    # 内部工具
    # --------

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = f"{self.base_url}{path}"
        resp = self._session.request(method, url, timeout=self.timeout, **kwargs)
        if not (200 <= resp.status_code < 300):
            try:
                body = resp.json()
            except Exception:  # pragma: no cover - 容错
                body = resp.text
            raise MSSTApiError(
                f"MSST API request failed: {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )
        return resp

    # --------
    # 同步批量分离
    # --------

    def create_batch_sync(
        self,
        *,
        input_path: str,
        output_dir: str,
        model_name: str,
        model_type: Optional[str] = None,
        extract_instrumental: Optional[list[str]] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """同步执行一次批量分离（``mode=sync``）。

        参数：

        - ``input_path``: 输入目录路径，建议使用绝对路径，对应容器挂载的 ``/app/input`` 等目录；
        - ``output_dir``: 输出目录路径，建议使用绝对路径，对应 ``/app/results`` 等；
        - ``model_name``: 具体模型文件名，例如
          ``"melband_roformer_instvox_duality_v2.ckpt"``；
        - ``model_type``: 模型类别字符串，例如 ``"vocal_models"``；
        - ``extract_instrumental``: 输出音轨列表，例如
          ``["Vocals", "Instrumental"]``；为空或 ``None`` 时按模型默认导出全部音轨；
        - ``range_start`` / ``range_end``: 批量处理范围（1 开头、闭区间），例如
          ``range_start=1, range_end=6`` 表示只处理排序后前 6 个子目录/文件；
        - ``params``: 额外参数字典，直接透传给后端（例如 device_ids/use_tta 等），
          不再推荐把音轨选择放在其中。

        返回值：

        - 后端 JSON 响应（包含 ``task_id`` 和最终 ``status`` 等字段）。

        使用建议：

        - 通用台词场景：

          ``model_type="vocal_models"``,
          ``model_name="melband_roformer_instvox_duality_v2.ckpt"``;

        - 需要强化 ME/伴奏质量时：

          ``model_type="vocal_models"``,
          ``model_name="mel_band_roformer_instrumental_becruily.ckpt"``。
        """

        merged_params: Dict[str, Any] = dict(params or {})
        if range_start is not None:
            merged_params["range_start"] = range_start
        if range_end is not None:
            merged_params["range_end"] = range_end

        payload: Dict[str, Any] = {
            "input_path": input_path,
            "output_dir": output_dir,
            "model_name": model_name,
            "params": merged_params,
        }
        if model_type is not None:
            payload["model_type"] = model_type
        if extract_instrumental is not None:
            payload["extract_instrumental"] = extract_instrumental

        resp = self._request(
            "POST",
            "/api/v1/tasks/msst-batch",
            params={"mode": "sync"},
            json=payload,
        )
        return resp.json()

    # --------
    # SSE 流式批量分离
    # --------

    def create_batch_sse(
        self,
        *,
        input_path: str,
        output_dir: str,
        model_name: str,
        model_type: Optional[str] = None,
        extract_instrumental: Optional[list[str]] = None,
        range_start: Optional[int] = None,
        range_end: Optional[int] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Iterable[SSEEvent]:
        """以 SSE 事件流的方式执行一次批量分离（``mode=sse``）。

        返回一个可迭代对象，遍历时会逐条产出 :class:`SSEEvent`：

        - ``event``: ``"start"`` / ``"progress"`` / ``"completed"`` /
          ``"error"`` / ``None``；
        - ``data``: 后端发送的 JSON 数据。

        常见用法::

            client = MSSTApiClient()
            for ev in client.create_batch_sse(
                input_path="/TTD/04_应用/msst/INPUT_BATCH",
                output_dir="/TTD/04_应用/msst/OUTPUT",
                model_type="vocal_models",
                model_name="mel_band_roformer_instrumental_becruily.ckpt",
            ):
                if ev.event == "progress":
                    # 进度更新
                    print(ev.data.get("processed_files"), ev.data.get("total_files"))
                elif ev.event == "error":
                    # 错误处理
                    raise RuntimeError(ev.data)
        """

        merged_params: Dict[str, Any] = dict(params or {})
        if range_start is not None:
            merged_params["range_start"] = range_start
        if range_end is not None:
            merged_params["range_end"] = range_end

        payload: Dict[str, Any] = {
            "input_path": input_path,
            "output_dir": output_dir,
            "model_name": model_name,
            "params": merged_params,
        }
        if model_type is not None:
            payload["model_type"] = model_type
        if extract_instrumental is not None:
            payload["extract_instrumental"] = extract_instrumental

        url = f"{self.base_url}/api/v1/tasks/msst-batch"
        resp = self._session.post(url, params={"mode": "sse"}, json=payload, stream=True, timeout=self.timeout)
        if not (200 <= resp.status_code < 300):
            try:
                body = resp.json()
            except Exception:  # pragma: no cover
                body = resp.text
            raise MSSTApiError(
                f"MSST API SSE request failed: {resp.status_code}",
                status_code=resp.status_code,
                body=body,
            )

        return self._iter_sse(resp)

    def _iter_sse(self, resp: requests.Response) -> Generator[SSEEvent, None, None]:
        """解析 text/event-stream 响应为 SSEEvent 流。

        简单解析规则：
        - 聚合连续的 `event:` / `data:` 行，遇到空行视为一个事件结束。
        - 仅支持单个 `data:`，其内容必须是 JSON。
        """

        event: Optional[str] = None
        data: Optional[Dict[str, Any]] = None

        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.strip()
            if not line:
                # 事件结束
                if data is not None:
                    yield SSEEvent(event=event, data=data)
                event = None
                data = None
                continue

            if line.startswith("event:"):
                event = line[len("event:") :].strip() or None
            elif line.startswith("data:"):
                payload = line[len("data:") :].strip()
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    # 非 JSON 数据也以原始字符串形式返回
                    data = {"raw": payload}

        # 流以非空状态结束时，补一个事件
        if data is not None:
            yield SSEEvent(event=event, data=data)

    # --------
    # 状态与结果查询
    # --------

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """查询任务状态。"""

        resp = self._request("GET", f"/api/v1/tasks/{task_id}")
        return resp.json()

    def get_task_result(self, task_id: str) -> Dict[str, Any]:
        """查询任务结果。"""

        resp = self._request("GET", f"/api/v1/tasks/{task_id}/result")
        return resp.json()


    def cancel_task(self, task_id: str) -> Dict[str, Any]:
        """取消（中断）一个 SSE 任务。

        服务端会尽可能保留并返回已生成的输出文件（status=canceled）。
        """

        resp = self._request("POST", f"/api/v1/tasks/{task_id}/cancel")
        return resp.json()

    # --------
    # 模型列表
    # --------

    def list_models(self, model_class: Optional[str] = None) -> Dict[str, Any]:
        """列出可用模型。

        :param model_class: 可选，按 ``models_info.json`` 中的 ``model_class``
            过滤，例如 ``"VR_Models"``、``"vocal_models"`` 等。
        :return: 后端返回的 JSON（``ModelListResponse``）。

        典型用法（为 agent 自动选择模型提供支持）::

            client = MSSTApiClient()
            models_resp = client.list_models(model_class="vocal_models")
            candidates = {
                "default_vocal": "melband_roformer_instvox_duality_v2.ckpt",
                "strong_me": "mel_band_roformer_instrumental_becruily.ckpt",
            }
            available = {m["id"] for m in models_resp["models"]}

            # 选择第一个既在推荐列表又在当前环境已安装的模型
            selected = None
            for key in ["default_vocal", "strong_me"]:
                if candidates[key] in available:
                    selected = candidates[key]
                    break

            # fall back：若推荐模型均不存在，可退回到任意 "vocal_models" 模型
            if selected is None and models_resp["models"]:
                selected = models_resp["models"][0]["id"]

        这样可以让上层逻辑在不同机器/模型安装状态下保持鲁棒的自动选择行为。
        """

        params: Dict[str, Any] = {}
        if model_class is not None:
            params["model_class"] = model_class
        resp = self._request("GET", "/api/v1/models", params=params)
        return resp.json()


__all__ = ["MSSTApiClient", "SSEEvent", "MSSTApiError"]
