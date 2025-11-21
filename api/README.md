# MSST-WebUI API 详细设计（方案 1：单进程 FastAPI + Gradio）

> 本文档对应项目根目录下的 `api.md`（API 核心计划），从**实现视角**细化模块划分、目录结构和运行时行为。实现代码全部放在 `api/` 目录内，避免修改上游项目核心文件。

---

## 1. 背景与目标

- **背景**
  - 上游项目提供基于 Gradio 的 WebUI（`webUI.py`、`webui/`）。
  - 当前 Docker 入口 `docker/entry.py` 已经采用：
    - `setup_webui()` 初始化环境与配置；
    - `webui.app.app(...)` 创建 Gradio Blocks；
    - `FastAPI()` + `gr.mount_gradio_app(...)` 在同一进程中承载 Gradio。
- **本设计的目标**
  - 在维持上述单进程架构的前提下，新增一套 **批量 MSST 分离 API**：
    - 与 Gradio WebUI 共存于同一进程，**共享同一份模型显存**；
    - 接口规范与行为与 `api.md` 中的核心计划保持一致；
    - 所有实现代码集中于 `api/` 目录，便于升级上游代码时减小冲突面。

---

## 2. 总体架构（运行时视图）

### 2.1 进程与组件

- 单一 Python 进程，主要组件：
  - **FastAPI 应用**：
    - 根路径 `/`：挂载 Gradio WebUI。
    - `/api/v1/...`：批量 MSST 分离 API（本设计的实现范围）。
  - **Gradio Blocks**：
    - 由 `webui.app.app(...)` 创建，内部持有模型、推理 pipeline 等。
  - **API 模块（`api/`）**：
    - 定义 Pydantic 模型、路由、服务层与任务管理逻辑。

### 2.2 docker/entry.py 中的集成方式

- `docker/entry.py` 中 `create_app()` 的目标形态：
  1. 调用 `setup_webui()` 完成配置/环境初始化；
  2. 调用 `webui.app.app(...)` 创建 `demo`（Gradio Blocks），并 `.queue()`；
  3. 创建 `fastapi_app = FastAPI()`；
  4. 调用 `gr.mount_gradio_app(fastapi_app, demo, path="/")`；
  5. 调用 `api` 目录中的入口函数，为 `fastapi_app` 注册 `/api/v1/...` 路由；
  6. 返回 `fastapi_app`，由 `uvicorn.run` 启动。

- **关键点**：
  - Gradio 和 API 共用同一 `fastapi_app` 与同一进程中的模型实例。
  - API 层不直接创建/管理模型，只通过现有 pipeline 或公共服务接口间接调用。

---

## 3. 目录与模块规划（`api/`）

> 以下为初始规划，后续实现时尽量遵守，不在上游目录散落新代码。

- `api/README.md`
  - 本文档，描述详细设计与约束。
- `api/app.py`
  - 提供 API 注册入口，例如：
    - `def register_routes(app: FastAPI) -> None:`
    - 或 `router = APIRouter()` 供 `app.include_router` 使用。
  - 内部组织：
    - 只负责组合路由与依赖注入，不写业务细节。
- `api/models.py`
  - 定义所有与 HTTP 交互相关的 Pydantic 模型：
    - `TaskCreateRequest` / `TaskCreateResponse`；
    - `TaskStatusResponse`；
    - `TaskResultResponse` / `FileResult`；
    - 公共错误响应模型等。
  - 与 `api.md` 第 3、4 节中数据结构保持一致。
- `api/services/tasks.py`
  - 核心领域逻辑：
    - 创建任务（扫描输入路径、生成文件列表、持久化 Task/TaskResult 雏形）。
    - 更新任务状态、进度。
    - 触发后台执行（同步/异步方式见第 5 章）。
    - 汇总并返回结果数据结构。
- `api/storage/__init__.py`
  - 定义统一的存储抽象（接口/基类），屏蔽具体后端差异。
- `api/storage/memory.py`
  - 进程内内存存储实现，用于开发环境和轻量使用场景：
    - `tasks: Dict[str, Task]`
    - `task_results: Dict[str, TaskResult]`
  - 不提供跨进程/重启后的持久化能力。
- `api/storage/sqlite.py`（推荐的可持久化实现）
  - 基于 SQLite 的简单文件数据库存储：
    - 使用单一 `.db` 文件，适合 Docker 卷挂载和备份。
    - 在容器重启后保留任务元数据和结果记录（具体保留策略可由配置控制）。
  - 通过统一存储抽象暴露给上层，不改变服务层/路由层接口。
- `api/config.py`
  - 一些可调参数：最大并发、任务超时时间、结果保留数量、默认存储后端类型（memory/sqlite）等。

---

## 4. API 层设计（对照 api.md）

### 4.1 路由布局

- 基础前缀：`/api/v1`
- 路由：
  - `POST /tasks/msst-batch`
    - Query 参数 `mode`：
      - `sync`（默认）：同步阻塞调用，返回 JSON；
      - `sse`：返回 `text/event-stream`，标准 SSE 事件流，渐进式推送进度。
  - `GET /tasks/{task_id}`
  - `GET /tasks/{task_id}/result`

### 4.2 模型定义草案

> 仅在文档中描述字段，实际实现时在 `api/models.py` 使用 Pydantic 定义。

- `TaskCreateRequest`
  - `input_path: str`
  - `output_dir: str`
  - `model_type: str`
  - `model_name: str`
  - `params: Optional[Dict[str, Any]]`
- `TaskCreateResponse`
  - `task_id: str`
  - `status: str`  （`pending` / `queued`）
  - `message: Optional[str]`
- `TaskStatusResponse`
  - `task_id: str`
  - `status: str`  （`pending` / `running` / `success` / `failed` / `canceled` / `success_with_errors`）
  - `progress: Optional[float]`
  - `total_files: Optional[int]`
  - `processed_files: Optional[int]`
  - `error: Optional[str]`
  - `created_at: datetime`
  - `updated_at: datetime`
- `FileResult`
  - `input_file: str`
  - `output_files: List[str]`
  - `status: str`
  - `error: Optional[str]`
- `TaskResultResponse`
  - `task_id: str`
  - `status: str`
  - `files: List[FileResult]`

---

## 5. 任务执行模型（首版建议）

### 5.1 异步任务策略

- 为简化首版实现，采用 **进程内后台线程/协程队列** 模型（后续可进一步提炼）：
  - `POST /tasks/msst-batch`：
    - 在请求线程中完成参数校验与扫描输入路径，生成文件列表。
    - 创建 `Task` / `TaskResult` 初始记录（通过统一存储抽象写入后端：内存或 SQLite）。
    - 将任务 ID 推入内部队列，后台 worker 消费。
  - 后台 worker：
    - 从队列中取出任务 ID；
    - 逐个文件调用现有 MSST 分离逻辑；
    - 更新任务的 `status` / `progress` / `processed_files`（写回所选存储后端）；
    - 填充 `TaskResult.files`。

### 5.2 并发与资源控制

- 首版控制策略：
  - 每个任务内部 **串行** 或小规模并发处理文件（例如固定线程池大小）；
  - 全局限制同时运行的任务数（例如队列长度 + worker 数量配置）；
  - 具体参数在 `api/config.py` 中配置，不暴露给 HTTP 调用方。

> 后续如有需要，可演进为使用 `asyncio` 队列或 external job queue，但首版仅依赖 Python 标准库即可。

---

## 6. 与现有 MSST-WebUI 的集成点

- 不直接在 API 中创建/管理模型实例，而是：
  - 通过与 Gradio 共用的 pipeline 函数或模块调用现有推理逻辑；
  - 或重用已经抽出来的“分离函数”（如果后续发现有现成模块，可在实现期进一步绑定）。
- 约束：
  - API 不改变原有 WebUI 的工作目录结构：
    - 仍然使用 `input/`、`results/`、`cache/` 等目录；
    - 任务的 `output_dir` 应与现有结果目录策略兼容，避免破坏 WebUI 行为。

---

## 7. 错误处理与日志

- 统一由 FastAPI 处理 HTTP 层异常，API 业务逻辑内部：
  - 捕获预期错误（路径不存在、权限问题等），映射为 `400` 或 `404`；
  - 对于非预期错误，返回 `500`，并通过项目现有的 `logger` 打印完整栈信息（实现时调用 `logger.exception` 或等价机制）。
- 错误响应模型使用统一结构（与 `api.md` 第 6 节一致），例如：
  - `error_code: Optional[str]`
  - `error_message: str`

---

## 8. 部署与启动流程

- Docker 中：
  - 仍由 `docker/entry.py` 作为容器入口；
  - `create_app()` 中在挂载 Gradio 后，调用 `api.app.register_routes(fastapi_app)`（或等价方式）；
  - 对外只暴露一个 HTTP 端口（即当前已经使用的端口），既可访问 WebUI，又可访问 `/api/v1/...`。

---

## 9. 后续可演进方向（非本轮实现范围）

- 引入持久化任务存储（SQLite/Redis 等），支持容器重启后的任务恢复与查询。
- 增加鉴权（例如 API Key）与简单限流，适配对外开放场景。
- 支持上传文件/外部存储 URL 作为输入源。
- 将 Gradio 前端改造成 API 客户端，从而可以在不同进程/机器访问同一后端 API 服务。

> 本详细设计与根目录 `api.md` 一起构成本项目 API 的设计基线。实现阶段应优先遵循本文件的模块边界与依赖关系，如需重大偏离，建议先在此文档中更新设计后再实施。

---

## 10. 项目级处理支持

### 10.1 业务场景支持

API 完全支持项目级的批量处理需求，典型场景：

- **输入目录**：`/TTD/00_项目/xxx/input` - 剧集文件目录
- **输出目录**：`/TTD/00_项目/xxx/03_Stem/xxx` - 分离结果目录

### 10.2 路径灵活性

- **任意路径支持**：API 支持容器内可访问的任意绝对路径
- **项目级挂载**：通过 Docker 挂载 `/TTD:/TTD` 支持整个项目目录结构
- **向后兼容**：保持原有的 `/app/input` 和 `/app/results` 挂载

### 10.3 使用示例

```python
from api.client import MSSTApiClient

client = MSSTApiClient()

# 项目级处理
resp = client.create_batch_sync(
    input_path='/TTD/00_项目/demo_project/input',           # 项目输入目录
    output_dir='/TTD/00_项目/demo_project/03_Stem/demo_project',  # 项目输出目录
    model_type='vocal_models',
    model_name='melband_roformer_instvox_duality_v2.ckpt',
    extract_instrumental=['Vocals', 'Instrumental'],
    params={'device_ids': ['cuda:0'], 'use_tta': False},
)

# 获取结果
task_id = resp['task_id']
result = client.get_task_result(task_id)
for file_result in result['files']:
    print(f"输入: {file_result['input_file']}")
    for output_file in file_result['output_files']:
        print(f"  输出: {output_file}")
```

### 10.4 输出结构

每个任务会在指定的输出目录下创建独立的任务子目录：

```text
/TTD/00_项目/demo_project/03_Stem/demo_project/
└── task_20251121063959815598/    # 任务独立目录
    ├── episode1_Vocals.wav
    ├── episode1_Instrumental.wav
    ├── episode2_Vocals.wav
    └── episode2_Instrumental.wav
```

### 10.5 文件支持

- **音频格式**：`.wav`, `.mp3`, `.flac`, `.m4a`, `.aac`, `.ogg`
- **视频格式**：`.mp4`, `.avi`, `.mkv`, `.mov`, `.wmv`, `.flv`, `.webm`
- **自动过滤**：排除 NAS 系统文件（以 `@` 或 `.` 开头）

### 10.6 Range 参数

支持指定文件范围处理，适用于大型项目的分批处理：

```python
resp = client.create_batch_sync(
    input_path='/TTD/00_项目/demo_project/input',
    output_dir='/TTD/00_项目/demo_project/03_Stem/demo_project',
    range_start=1,    # 处理第1-5个文件
    range_end=5,
    # ... 其他参数
)
```

---

## 11. 超时与长任务支持

- **客户端（Python requests）**
  - `api/client.py` 中的 :class:`MSSTApiClient` 默认使用 `timeout=None`：
    - 适合长时间的批量分离任务和 SSE 事件流；
    - 如上游网关/代理对请求时长有限制，可在初始化时显式传入秒级超时。

- **FastAPI / uvicorn（容器内部）**
  - `docker/entry.py` 中通过 `uvicorn.run(app, host=..., port=..., log_level="info")` 启动服务，
    当前未显式配置请求级超时；
  - 推荐保持应用层不对单个请求设置额外超时限制，让长时间的推理或 SSE 由业务自行终止；
  - 如需自定义 uvicorn 参数（例如使用 CLI 或其他进程管理器），应避免将与请求生命周期强绑定的
    超时（如强制中断长连接）设置得过小，以免影响长任务和 SSE。

- **Caddy2 反向代理（建议配置）**
  - 若在 Caddy2 中通过 `reverse_proxy` 将外部请求转发到容器内的 `http://ttd-edge:8662`：
    - 应确保代理层的 `read_timeout` / `idle_timeout` 设置足够大或显式关闭，
      以适配长时间的下载流（包括 SSE 和长时间的 JSON 响应）；
    - 示例（仅为参考，需要根据实际 Caddyfile 位置合并）：

      ```caddyfile
      :80 {
          reverse_proxy /api/* ttd-edge:8662 {
              transport http {
                  # 对长时间推理 / SSE 友好的超时设置示例
                  read_timeout  0s       # 0 表示不限制读取超时
                  idle_timeout  0s       # 0 表示不限制空闲连接时间
              }
          }
      }
      ```

  - 实际部署时，可根据运维策略适当收紧上述超时，但建议：
    - 不要将 `read_timeout` / `idle_timeout` 设置为明显小于典型任务时长的值；
    - 对 SSE 场景，优先采用“长连接 + 心跳事件”的模式，而不是依赖代理层的短超时重连。
