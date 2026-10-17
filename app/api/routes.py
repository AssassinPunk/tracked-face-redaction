"""HTTP endpoints. Kept thin: they validate, call app.core, and shape responses."""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import __version__

router = APIRouter(prefix="/api")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness and model status",
    responses={503: {"model": HealthResponse, "description": "Model not loaded"}},
)
def health(request: Request) -> JSONResponse:
    loaded = request.app.state.detector is not None
    body = HealthResponse(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded,
        version=__version__,
    )
    return JSONResponse(body.model_dump(), status_code=200 if loaded else 503)