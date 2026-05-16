from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from src.serving.config import settings


class YOLOModelService:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model: YOLO | None = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}. "
                "Place best.pt in models/yolo26n_visdrone_baseline/ "
                "or set MODEL_PATH."
            )

        self.model = YOLO(str(self.model_path))

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, image, conf: float | None = None):
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        return self.model.predict(
            source=image,
            imgsz=settings.image_size,
            conf=conf if conf is not None else settings.confidence_threshold,
            iou=settings.iou_threshold,
            verbose=False,
        )

    @property
    def class_names(self) -> dict[int, str]:
        if self.model is None:
            return {}
        return dict(self.model.names)


model_service = YOLOModelService(settings.model_path)
