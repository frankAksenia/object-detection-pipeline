from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VisDrone YOLO API"
    app_version: str = "0.1.0"

    # default values can be overridden by environment variables or .env file
    model_path: Path = Path("models/yolo26n_visdrone_baseline/best.pt")
    image_size: int = 640
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.7

    skip_model_load: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


settings = Settings()
