from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO prediction.")

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--exist-ok", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(str(args.model))

    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(args.project),
        name=args.name,
        save=True,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
