from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskCreateRequest(BaseModel):
    input_path: str = Field(..., description="输入音频路径（文件或目录）")
    output_dir: str = Field(..., description="输出目录（服务端可写目录）")
    model_name: str = Field(..., description="MSST 模型名称，对应现有配置里的模型 key")
    model_type: Optional[str] = Field(None, description="模型类型，可选，用于校验或未来扩展")
    extract_instrumental: Optional[List[str]] = Field(
        default=None,
        description="选择输出的音轨列表，例如 ['Vocals', 'Instrumental']；为空或未提供时按模型默认导出全部音轨。",
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="其他可选参数，例如 device/output_format/use_tta 等（兼容旧版调用，不推荐继续放音轨选择）",
    )


class TaskCreateResponse(BaseModel):
    task_id: str
    status: str
    message: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    progress: Optional[float] = None
    total_files: Optional[int] = None
    processed_files: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class FileResult(BaseModel):
    input_file: str
    output_files: List[str] = Field(default_factory=list)
    status: str
    error: Optional[str] = None


class TaskResultResponse(BaseModel):
    task_id: str
    status: str
    files: List[FileResult] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error_code: Optional[str] = None
    error_message: str


class ModelInfo(BaseModel):
    """可用模型的信息摘要。

    为了兼容不同模型类型，仅对通用字段做强约束，其他字段放入 extra 中。
    """

    id: str = Field(..., description="模型在 models_info.json 中的 key")
    model_name: str
    model_class: str
    model_size: Optional[int] = None
    sha256: Optional[str] = None
    is_installed: Optional[bool] = None
    target_position: Optional[str] = None
    link: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict, description="模型特有的其他字段，例如 primary_stem、secondary_stem 等")


class ModelListResponse(BaseModel):
    models: List[ModelInfo] = Field(default_factory=list)
