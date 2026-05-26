from __future__ import annotations

import time
from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.serving.config import settings
from src.serving.model_loader import model_service
from src.serving.schemas import (
    Detection,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded(),
        model_path=str(settings.model_path),
    )


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    return ModelInfoResponse(
        model_path=str(settings.model_path),
        image_size=settings.image_size,
        confidence_threshold=settings.confidence_threshold,
        iou_threshold=settings.iou_threshold,
        classes=model_service.class_names,
    )


@router.post("/predict/image", response_model=PredictionResponse)
async def predict_image(
    file: Annotated[UploadFile, File(...)],
) -> PredictionResponse:
    if not model_service.is_loaded():
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be an image.",
        )

    image_bytes = await file.read()

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read image: {exc}",
        ) from exc

    start = time.perf_counter()
    results = model_service.predict(image)
    inference_ms = (time.perf_counter() - start) * 1000

    result = results[0]
    detections: list[Detection] = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            bbox_xyxy = [float(value) for value in box.xyxy[0].tolist()]

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=model_service.class_names.get(class_id, "unknown"),
                    confidence=confidence,
                    bbox_xyxy=bbox_xyxy,
                )
            )

    return PredictionResponse(
        filename=file.filename or "uploaded_image",
        model_path=str(settings.model_path),
        image_width=image.width,
        image_height=image.height,
        inference_ms=round(inference_ms, 2),
        detections_count=len(detections),
        detections=detections,
    )
