from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model.")

    parser.add_argument("--data", type=Path, required=True,
                        help="Path to dataset YAML.")
    parser.add_argument("--model", type=str,
                        default="yolo26n.pt", help="YOLO model checkpoint.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--exist-ok", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)

    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
