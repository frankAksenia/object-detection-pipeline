from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model.")

    parser.add_argument("--model", type=Path, required=True, help="Path to trained model.")
    parser.add_argument("--data", type=Path, required=True, help="Path to dataset YAML.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--exist-ok", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(str(args.model))

    metrics = model.val(
        data=str(args.data),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
    )

    metrics_dict = {
        "model_path": str(args.model),
        "data": str(args.data),
        "imgsz": args.imgsz,
        "batch": args.batch,
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
