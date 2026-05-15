from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a YOLO model.")

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--format", type=str, default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(str(args.model))

    model.export(
        format=args.format,
        imgsz=args.imgsz,
        device=args.device,
    )


if __name__ == "__main__":
    main()
