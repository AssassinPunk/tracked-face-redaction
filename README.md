# Tracked Face Redaction

![CI](https://github.com/AssassinPunk/tracked-face-redaction/actions/workflows/ci.yml/badge.svg)

Detect and anonymise faces in images through a REST API. Video support with
detect-every-N-frames tracking is in progress; its speed / missed-face
trade-off will be measured rather than assumed.

> **Status:** image redaction API complete (Milestones 1-4).
> Video, tracking, benchmarking, web UI and deployment are in progress.

## Features

**Implemented**
- Face detection with **YuNet** (OpenCV), including large phone photos
- Three anonymisation modes: **pixelate** (default), **blur**, **mask**
- Default strength chosen **by measurement**: redacted output is re-scanned
  and must contain no detectable faces ([experiment log](docs/experiments.md))
- **FastAPI** REST API with upload validation and generated docs at `/docs`
- Images processed entirely in memory; output carries no EXIF/GPS metadata
- 61 automated tests, run on every push by GitHub Actions

**Roadmap**
- [ ] Video redaction
- [ ] Detect every N frames, track in between (Norfair)
- [ ] Speed vs. recall benchmark on hand-annotated frames
- [ ] Web interface
- [ ] Docker image and public deployment

## Quick start

Requires **Python 3.11** (Norfair 2.3.0 needs NumPy < 2, which pins
compatible Python and OpenCV versions).

```bash
git clone https://github.com/AssassinPunk/tracked-face-redaction.git
cd tracked-face-redaction
python3.11 -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/download_model.py   # fetches YuNet weights (MIT), SHA-256 verified
python -m pytest                    # 61 tests
uvicorn app.main:app --reload       # API at http://127.0.0.1:8000/docs
```

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | Liveness and model status |
| `POST` | `/api/redact/image` | Upload an image, receive it with every face redacted |
| `GET` | `/docs` | Interactive OpenAPI documentation |

### `POST /api/redact/image`

**Body:** `multipart/form-data` with a `file` field (JPEG, PNG or WebP).
The response uses the same format as the upload.

| Query parameter | Type | Default | Meaning |
|---|---|---|---|
| `mode` | `pixelate` \| `blur` \| `mask` | `pixelate` | How faces are hidden |
| `strength` | float in (0, 1] | measured per mode | Pixel block or blur kernel, as a fraction of face size |

| Status | When |
|---|---|
| `200` | Redacted image returned |
| `400` | File is not a decodable image (even if it claims to be one) |
| `413` | File over 10 MB, or more than 40 megapixels |
| `415` | Unsupported content type |
| `422` | Invalid `mode` or `strength` |
| `503` | Detection model not loaded |

**Response headers:** `X-Faces-Detected` (count), `Cache-Control: no-store`,
and a fixed download filename (`redacted.jpg`; user filenames are never echoed).

**Example (bash / macOS / Linux):**
```bash
curl -X POST "http://127.0.0.1:8000/api/redact/image?mode=pixelate" \
  -F "file=@photo.jpg;type=image/jpeg" -o redacted.jpg -D -
```

**Example (Windows PowerShell):**
```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/redact/image?mode=pixelate" -F "file=@photo.jpg;type=image/jpeg" -o redacted.jpg -D -
```

## Design decisions

- **Measured defaults, not guesses.** The first defaults (blur 0.5, pixelate
  0.12) left 21 of 24 and 17 of 24 faces detectable after redaction. A strength
  sweep found pixelation at 0.25 left none; the default is 0.3 for margin. Blur
  never reached zero even at maximum strength, because Gaussian blur is a
  low-pass filter that keeps the coarse face layout detectors use. A regression
  test locks the guarantee in.
- **Recall over precision.** Raising the confidence threshold from 0.6 to 0.9
  dropped 37.5% of real faces with no speed gain, so the threshold stays low
  and is not client-adjustable.
- **Model loaded once at startup**, shared across requests. Because
  `setInputSize()` and `detect()` mutate the shared network, they run under a
  lock; a test runs 40 detections across 8 threads to verify it.
- **Validation cheapest-first:** declared type (415), bounded read (413),
  actual decode (400), pixel count (413).
- **No database, no stored images.** Images are processed in memory and
  returned; nothing persists.

## Testing

```bash
python -m pytest
```
Unit tests use synthetic images. Integration tests use a public-domain NASA
crew portrait with a known face count (see [tests/fixtures](tests/fixtures)),
including an end-to-end check that the API's output has no detectable faces.

## Project structure
```
app/
  api/routes.py         HTTP endpoints and validation
  core/detector.py      YuNet wrapper (thread-safe)
  core/anonymize.py     blur / pixelate / mask, box expansion and clipping
  core/image_pipeline.py  bytes in -> detect -> anonymise -> bytes out
  config.py             settings from environment variables
  main.py               app factory, startup model loading
scripts/                model download, demos, fixture builder
tests/                  unit, fixture and API tests
docs/experiments.md     measured results behind each decision
```

## Acknowledgements
- [YuNet](https://github.com/opencv/opencv_zoo) face detector (MIT), OpenCV Zoo
- Test fixture: Artemis II crew portrait, NASA / Josh Valcarcel, public domain

## Licence
MIT. See [LICENSE](LICENSE).