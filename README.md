# Tracked Face Redaction

![CI](https://github.com/AssassinPunk/tracked-face-redaction/actions/workflows/ci.yml/badge.svg)

Face detection and anonymisation for images and video. Faces are detected
every Nth video frame and tracked in between, and the speed / missed-face
trade-off of that choice is measured rather than assumed.

> **Status:** 🚧 In development. Milestone 1 (project setup) complete.

## Planned features
- Detect faces in images and video with **YuNet** (OpenCV)
- Anonymise faces by Gaussian blur, pixelation or solid mask
- Detect every N frames and track between them with **Norfair**
- REST API with **FastAPI** and auto-generated docs at `/docs`
- Reproducible benchmark of speed vs. recall for N = 1, 2, 5, 10, 20

## Tech stack
Python 3.11 · OpenCV · NumPy · Norfair · FastAPI · pytest · GitHub Actions · Docker

## Getting started
```bash
git clone https://github.com/AssassinPunk/tracked-face-redaction.git
cd tracked-face-redaction
python3.11 -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python scripts/verify_env.py      # should print ENVIRONMENT OK
python -m pytest
```

> **Python 3.11 is required.** Norfair 2.3.0 needs NumPy < 2, which pins
> the compatible Python and OpenCV versions.

## Project structure
```
app/            application code (core/ = image & video processing)
scripts/        developer utilities
tests/          automated tests
```

## Licence
MIT. See [LICENSE](LICENSE).