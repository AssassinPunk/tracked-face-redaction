"""HTTP endpoints. Kept thin: they validate, call app.core, and shape responses."""
from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from app import __version__
from app.core.anonymize import DEFAULT_MODE, Mode
from app.core.image_pipeline import decode_image, encode_image, redact_image

router = APIRouter(prefix="/api")

# Declared content type -> output extension. Output keeps the input format.
ALLOWED_IMAGE_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    detail: str


@router.get(
    "/health",
    tags=["system"],
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


@router.post(
    "/redact/image",
    tags=["redaction"],
    summary="Detect and anonymise every face in an image",
    response_class=Response,
    responses={
        200: {
            "description": "The redacted image, in the same format as the upload.",
            "content": {t: {} for t in ALLOWED_IMAGE_TYPES},
        },
        400: {"model": ErrorResponse, "description": "File is not a decodable image"},
        413: {"model": ErrorResponse, "description": "File or pixel count too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type"},
        503: {"model": ErrorResponse, "description": "Model not loaded"},
    },
)
def redact_image_endpoint(  # plain `def`: CPU-bound work runs in the thread pool
    request: Request,
    file: UploadFile = File(..., description="JPEG, PNG or WebP image."),
    mode: Mode = Query(DEFAULT_MODE, description="How to hide faces. Pixelate is the "
                       "measured default; blur preserves context but is weaker."),
    strength: float | None = Query(None, gt=0, le=1, description="Fraction of face size "
                                   "used as blur kernel or pixel block. Omit for the "
                                   "measured default."),
) -> Response:
    detector = request.app.state.detector
    if detector is None:
        raise HTTPException(503, "Face detection model is not loaded.")
    settings = request.app.state.settings

    # 1. Declared type (cheap, but clients can lie).
    ext = ALLOWED_IMAGE_TYPES.get(file.content_type or "")
    if ext is None:
        raise HTTPException(415, f"Unsupported file type. Allowed: {', '.join(ALLOWED_IMAGE_TYPES)}.")

    # 2. Size: read at most limit + 1 bytes, so processing memory is bounded.
    data = file.file.read(settings.max_image_bytes + 1)
    if len(data) > settings.max_image_bytes:
        mb = settings.max_image_bytes / (1024 * 1024)
        raise HTTPException(413, f"Image is larger than the {mb:g} MB limit.")

    # 3. Actual content: the bytes must really decode as an image.
    try:
        image = decode_image(data)
    except ValueError:
        raise HTTPException(400, "File could not be decoded as an image.") from None

    # 4. Pixel count: guards against tiny files that decode to huge images.
    height, width = image.shape[:2]
    if height * width > settings.max_image_pixels:
        raise HTTPException(413, f"Image exceeds the {settings.max_image_pixels:,}-pixel limit.")

    result = redact_image(image, detector, mode=mode, strength=strength)
    return Response(
        content=encode_image(result.image, ext),
        media_type=file.content_type,
        headers={
            "X-Faces-Detected": str(len(result.faces)),
            "Content-Disposition": f'attachment; filename="redacted{ext}"',  # never echo user input
            "Cache-Control": "no-store",  # don't let browsers or proxies keep a copy
        },
    )