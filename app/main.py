"""FastAPI application wiring. All image processing lives in app/core."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI

from app import __version__
from app.api.routes import router
from app.config import Settings
from app.core.detector import FaceDetector

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Runs ONCE at startup: load the model here, never per request.
        app.state.settings = settings
        try:
            app.state.detector = FaceDetector(model_path=settings.model_path)
            logger.info("YuNet model loaded from %s", settings.model_path)
        except (FileNotFoundError, cv2.error) as exc:
            # Stay up and report 503 from /api/health instead of crashing,
            # so the platform sees an honest "not ready" signal.
            app.state.detector = None
            logger.error("Model failed to load: %s", exc)  # details in logs only
        yield  # the server handles requests while paused here

    app = FastAPI(
        title="Tracked Face Redaction",
        version=__version__,
        description="Detect and anonymise faces in images and video.",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()  # what `uvicorn app.main:app` serves