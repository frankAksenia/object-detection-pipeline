from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model.")

    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to training config YAML. If provided, config values are used as defaults.",
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=False,
        help="Path to dataset YAML.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="YOLO model checkpoint.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--project", type=Path, required=False)
    parser.add_argument("--name", type=str, required=False)
    parser.add_argument("--exist-ok", action="store_true")

    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--mosaic", type=float, default=None)
    parser.add_argument("--close-mosaic", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--fliplr", type=float, default=None)
    parser.add_argument("--flipud", type=float, default=None)

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

    augmentation = config.get("augmentation", {})

    model_name = get_value(args.model, config, "model", "yolo26n.pt")
    data = get_value(args.data, config, "data")
    epochs = get_value(args.epochs, config, "epochs", 50)
    imgsz = get_value(args.imgsz, config, "imgsz", 640)
    batch = get_value(args.batch, config, "batch", 16)
    device = get_value(args.device, config, "device", "0")
    project = get_value(args.project, config, "project")
    name = get_value(args.name, config, "name")
    workers = get_value(args.workers, config, "workers", 2)
    patience = get_value(args.patience, config, "patience", 30)

    mosaic = get_value(args.mosaic, augmentation, "mosaic", 1.0)
    close_mosaic = get_value(args.close_mosaic, augmentation, "close_mosaic", 10)
    scale = get_value(args.scale, augmentation, "scale", 0.5)
    fliplr = get_value(args.fliplr, augmentation, "fliplr", 0.5)
    flipud = get_value(args.flipud, augmentation, "flipud", 0.0)

    if data is None:
        raise ValueError("Missing required value: --data or config field 'data'.")

    if project is None:
        raise ValueError("Missing required value: --project or config field 'project'.")

    if name is None:
        raise ValueError("Missing required value: --name or config field 'name'.")

    model = YOLO(str(model_name))

    model.train(
        data=str(data),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=str(device),
        project=str(project),
        name=str(name),
        exist_ok=args.exist_ok,
        workers=int(workers),
        patience=int(patience),
        mosaic=float(mosaic),
        close_mosaic=int(close_mosaic),
        scale=float(scale),
        fliplr=float(fliplr),
        flipud=float(flipud),
    )


if __name__ == "__main__":
    main()


#  Colab usage example:
#
#     python src/training/train.py \
#   --data /content/drive/MyDrive/mlops-visdrone/data/processed/visdrone-yolo/visdrone.yaml \
#   --model yolo26n.pt \
#   --epochs 50 \
#   --imgsz 640 \
#   --batch 16 \
#   --device 0 \
#   --project /content/drive/MyDrive/mlops-visdrone/runs \
#   --name yolo26n_img640_baseline \
#   --exist-ok
#
# or with config YAML:
#
#     python src/training/train.py \
#   --config configs/training/yolo26n_img960_e100.yaml