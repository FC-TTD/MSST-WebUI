import os

# 默认使用 SQLite 作为任务存储后端
DEFAULT_STORAGE_BACKEND = os.environ.get("MSST_API_STORAGE_BACKEND", "sqlite")

# SQLite 数据库文件路径（可通过环境变量覆盖）
DEFAULT_DB_PATH = os.environ.get(
    "MSST_API_DB_PATH",
    os.path.join("data", "api_tasks.db"),
)
