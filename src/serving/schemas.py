from __future__ import annotations

from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[float] = Field(
        description="Bounding box in [x1, y1, x2, y2] pixel coordinates."
    )


class PredictionResponse(BaseModel):
    filename: str
    model_path: str
    image_width: int
    image_height: int
    inference_ms: float
    detections_count: int
    detections: list[Detection]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


class ModelInfoResponse(BaseModel):
    model_path: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    classes: dict[int, str]
