# VisDrone YOLO Object Detection MLOps Pipeline

End-to-end computer vision pipeline for drone-based object detection using **YOLO**, **VisDrone2019-DET**, **FastAPI**, **Docker**, **GitHub Actions**, and reproducible training/evaluation workflows.

The project includes:

- Dataset download and conversion from VisDrone format to YOLO format
- Dataset quality/statistics analysis
- YOLO training in Google Colab with GPU
- Model evaluation with mAP, precision, recall, confusion matrix, and PR/F1 curves
- FastAPI inference service
- Dockerized API deployment
- Unit tests and CI with GitHub Actions
- Makefile-based development workflow

---

## Table of contents

- [Project overview](#project-overview)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Dataset](#dataset)
- [Dataset conversion](#dataset-conversion)
- [Dataset analysis](#dataset-analysis)
- [Training](#training)
- [FastAPI inference service](#fastapi-inference-service)
- [Docker usage](#docker-usage)
- [Development workflow](#development-workflow)
- [Makefile commands](#makefile-commands)
- [Testing strategy](#testing-strategy)
- [Continuous integration](#continuous-integration)
- [Reproducibility](#reproducibility)
- [Environment files](#environment-files)

---

## Project overview

This project trains and serves a YOLO object detection model on the **VisDrone2019-DET** dataset.

The pipeline includes data preparation, training, evaluation, API serving, testing, Dockerization, and CI.

The detection classes are:

| Class ID | Class name |
|---:|---|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |

---

## Architecture

```text
Raw VisDrone Dataset
        ↓
Download Script
        ↓
VisDrone → YOLO Conversion
        ↓
Dataset Statistics / Validation
        ↓
YOLO Training in Google Colab GPU
        ↓
Evaluation: mAP, Precision, Recall, PR Curve, Confusion Matrix
        ↓
Best Model Checkpoint
        ↓
FastAPI Inference Service
        ↓
Docker Container
        ↓
CI / Tests / Reproducible Development Workflow
```

---

## Tech stack

| Area | Tools |
|---|---|
| Computer vision | YOLO, Ultralytics, OpenCV, Pillow |
| Deep learning | PyTorch |
| Dataset | VisDrone2019-DET |
| Training environment | Google Colab GPU |
| API serving | FastAPI, Uvicorn |
| Packaging | Docker, Docker Compose |
| Testing | Pytest |
| Code quality | Ruff, Black |
| CI/CD | GitHub Actions |
| Experiment config | YAML |

---

## Dataset

This project uses the **VisDrone2019-DET** dataset.

The original VisDrone detection annotations use this format:

```text
bbox_left,bbox_top,bbox_width,bbox_height,score,object_category,truncation,occlusion
```

The project converts those annotations into standard YOLO format:

```text
class_id x_center y_center width height
```

where all bounding box coordinates are normalized to `[0, 1]`.

---

## Dataset conversion

The conversion script is located at:

```text
src/data/convert_visdrone_to_yolo.py
```

It performs the following steps:

- Reads VisDrone annotation files
- Skips ignored/invalid regions where `score == 0`
- Converts raw VisDrone class IDs `1–10` to YOLO class IDs `0–9`
- Converts absolute top-left `xywh` boxes to normalized YOLO center `xywh`
- Creates YOLO-compatible folders:

```text
images/train
images/val
images/test
labels/train
labels/val
labels/test
```

- Writes a dataset YAML file:

```text
visdrone.yaml
```

Example Colab command:

```bash
python src/data/convert_visdrone_to_yolo.py \
  --raw-root /content/drive/MyDrive/mlops-visdrone/data/raw/visdrone \
  --output-root /content/drive/MyDrive/mlops-visdrone/data/processed/visdrone-yolo \
  --splits train val test
```

Expected output:

```text
/content/drive/MyDrive/mlops-visdrone/data/processed/visdrone-yolo/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── visdrone.yaml
```

---

## Dataset analysis

Dataset statistics are generated with:

```bash
python src/data/dataset_stats.py \
  --labels-dir /content/drive/MyDrive/mlops-visdrone/data/processed/visdrone-yolo/labels/train \
  --output /content/drive/MyDrive/mlops-visdrone/reports/dataset_stats_train.json
```


---

## Training

Training is performed in **Google Colab** using GPU.

The main Colab notebook is:

```text
notebooks/01_visdrone_yolo_colab_pipeline.ipynb
```

The training script is:

```text
src/training/train.py
```

It supports both direct CLI arguments and YAML config files.

Example training command:

```bash
python src/training/train.py \
  --config configs/training/yolo26n_img960_e100.yaml \
  --exist-ok
```
---

## FastAPI inference service

The project includes a FastAPI service for image inference.

Source files:

```text
src/serving/
├── app.py
├── config.py
├── model_loader.py
├── routes.py
└── schemas.py
```

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/model/info` | GET | Model metadata and class names |
| `/predict/image` | POST | Upload image and receive detections |
| `/metrics` | GET | Prometheus-compatible metrics, if enabled |

### Model path

The API expects the trained model at:

```text
models/yolo26n_visdrone_baseline/best.pt
```

The model file is not committed to GitHub.

You can also override the path with an environment variable:

```bash
export VISDRONE_MODEL_PATH=/absolute/path/to/best.pt
```

### Run locally

```bash
make api
```

or:

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

### Health check

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/yolo26n_visdrone_baseline/best.pt"
}
```

### Image prediction

```bash
curl -X POST "http://localhost:8000/predict/image" \
  -F "file=@sample.jpg"
```

Example response:

```json
{
  "filename": "sample.jpg",
  "model_path": "models/yolo26n_visdrone_baseline/best.pt",
  "image_width": 1920,
  "image_height": 1080,
  "inference_ms": 52.31,
  "detections_count": 3,
  "detections": [
    {
      "class_id": 3,
      "class_name": "car",
      "confidence": 0.81,
      "bbox_xyxy": [103.2, 44.1, 189.7, 98.5]
    }
  ]
}
```

### Postman usage

Use:

```text
POST http://localhost:8000/predict/image
```

In Postman:

```text
Body → form-data
```

Add:

| Key | Type | Value |
|---|---|---|
| file | File | sample image |

Do not manually set the `Content-Type` header. Postman will set `multipart/form-data` automatically.

---

## Docker usage

The API can be run with Docker Compose.

Build the image:

```bash
make docker-build
```

Start the service:

```bash
make docker-up
```

Stop the service:

```bash
make docker-down
```

Or directly:

```bash
docker compose up --build
```

The Docker service mounts local model artifacts from:

```text
./models
```

Make sure your model exists locally:

```text
models/yolo26n_visdrone_baseline/best.pt
```

---

## Development workflow

Install development dependencies:

```bash
make install-dev
```

Format code:

```bash
make format
```

Run lint checks:

```bash
make lint
```

Run tests:

```bash
make test
```

Run all checks:

```bash
make check
```

---

## Makefile commands

| Command | Description |
|---|---|
| `make install` | Install API/runtime dependencies |
| `make install-dev` | Install development and testing dependencies |
| `make format` | Auto-format code with Ruff and Black |
| `make lint` | Run Ruff and Black checks |
| `make test` | Run Pytest unit tests |
| `make check` | Run linting and tests |
| `make api` | Start the FastAPI inference service locally |
| `make docker-build` | Build the Docker image |
| `make docker-up` | Start the API with Docker Compose |
| `make docker-down` | Stop Docker Compose services |
| `make train` | Train YOLO using a config file |
| `make evaluate` | Evaluate a trained YOLO model |
| `make dataset-stats` | Generate dataset statistics |
| `make clean` | Remove local Python/test caches |

---

## Testing strategy

The test suite is designed to be lightweight and deterministic.

Tests do not require:

- GPU
- full VisDrone dataset
- trained model checkpoint

Current test coverage includes:

| Test area | Purpose |
|---|---|
| FastAPI health endpoint | Verifies API startup behavior |
| Dataset conversion | Verifies VisDrone annotations are converted to YOLO format correctly |
| Ignored regions | Verifies invalid/ignored VisDrone boxes are skipped |
| Dataset statistics | Verifies class counts, empty label files, and quality checks |

The app supports skipping model loading in CI:

```bash
VISDRONE_SKIP_MODEL_LOAD=true
```

Run tests:

```bash
pytest tests -q
```

---

## Continuous integration

This project uses GitHub Actions for CI.

Workflow file:

```text
.github/workflows/test.yml
```

The CI pipeline runs on push and pull request.

It checks:

- Dependency installation
- Ruff linting
- Black formatting
- Pytest unit tests
- FastAPI health endpoint
- Dataset conversion utilities
- Dataset statistics utilities

Training is intentionally not run in CI because full VisDrone training requires a GPU and large dataset artifacts.

CI workflow:

```text
push / pull request
        ↓
checkout repo
        ↓
setup Python 3.12
        ↓
install dependencies
        ↓
ruff check
        ↓
black --check
        ↓
pytest
```

---

## Reproducibility

The project separates source code, configs, notebooks, reports, and large artifacts.

Tracked in Git:

- source code under `src/`
- training configs under `configs/`
- notebooks under `notebooks/`
- lightweight reports under `reports/`
- README assets under `docs/assets/`
- CI workflow under `.github/workflows/`
- dependency files
- Makefile

Not tracked in Git:

- raw datasets
- processed datasets
- YOLO run folders
- model checkpoints
- prediction outputs
- local environment files

Large artifacts such as `best.pt`, datasets, and full prediction outputs should be stored externally, for example in Google Drive or cloud storage.

---

## Environment files

Example `.env`:

```env
MODEL_PATH=models/yolo26n_visdrone_baseline/best.pt
IMAGE_SIZE=640
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.7
SKIP_MODEL_LOAD=false
```

Commit `.env.example`, but do not commit `.env`.

---
