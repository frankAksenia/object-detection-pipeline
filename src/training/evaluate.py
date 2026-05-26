from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model.")

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to training/evaluation config YAML.",
    )

    parser.add_argument(
        "--model", type=Path, required=False, help="Path to trained model."
    )
    parser.add_argument(
        "--data", type=Path, required=False, help="Path to dataset YAML."
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=Path, required=False)
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--exist-ok", action="store_true")

    return parser.parse_args()


def load_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None:
        return {}

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    return config


def get_value(
    cli_value: Any,
    config: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """
    Priority:
    1. CLI argument
    2. YAML config value
    3. default
    """
    if cli_value is not None:
        return cli_value

    if key in config:
        return config[key]

    return default


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    model_path = get_value(args.model, config, "model_path")
    data = get_value(args.data, config, "data")
    imgsz = get_value(args.imgsz, config, "imgsz", 640)
    batch = get_value(args.batch, config, "batch", 16)
    device = get_value(args.device, config, "device", "0")
    project = get_value(args.project, config, "eval_project", config.get("project"))
    name = get_value(args.name, config, "eval_name", config.get("name"))

    if model_path is None:
        raise ValueError(
            "Missing required value: --model or config field 'model_path'."
        )

    if data is None:
        raise ValueError("Missing required value: --data or config field 'data'.")

    if project is None:
        raise ValueError(
            "Missing required value: --project or config field 'eval_project'."
        )

    if name is None:
        raise ValueError("Missing required value: --name or config field 'eval_name'.")

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data),
        imgsz=int(imgsz),
        batch=int(batch),
        device=str(device),
        project=str(project),
        name=str(name),
        exist_ok=args.exist_ok,
    )

    metrics_dict = {
        "experiment_name": str(config.get("name", name)),
        "model_path": str(model_path),
        "data": str(data),
        "imgsz": int(imgsz),
        "batch": int(batch),
        "mAP50_95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "mAP75": float(metrics.box.map75),
        "mean_precision": float(metrics.box.mp),
        "mean_recall": float(metrics.box.mr),
    }

    print(json.dumps(metrics_dict, indent=2))

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(metrics_dict, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()


#  Colab usage example:
#
# python src/training/evaluate.py \
#   --model /content/drive/MyDrive/mlops-visdrone/
#           runs/yolo26n_img640_baseline/weights/best.pt \
#   --data /content/drive/MyDrive/mlops-visdrone/
#           data/processed/visdrone-yolo/visdrone.yaml \
#   --imgsz 640 \
#   --batch 16 \
#   --device 0 \
#   --project /content/drive/MyDrive/
#           mlops-visdrone/evaluation \
#   --name yolo26n_img640_baseline_val \
#   --metrics-output /content/drive/MyDrive/
#       mlops-visdrone/reports/metrics/
#       img640_baseline.json \
#   --exist-ok
#
# or with config YAML:
#
#     python src/training/evaluate.py \
#   --config configs/training/yolo26n_img960_e100.yaml \
#   --metrics-output /content/drive/MyDrive/
#       mlops-visdrone/reports/metrics/yolo26n_img960_e100.json \
#   --exist-ok
