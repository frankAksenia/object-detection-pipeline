from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.serving.config import settings
from src.serving.model_loader import model_service
from src.serving.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not settings.skip_model_load:
        model_service.load()
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="FastAPI service for YOLO-based VisDrone object detection.",
    lifespan=lifespan,
)

app.include_router(router)
