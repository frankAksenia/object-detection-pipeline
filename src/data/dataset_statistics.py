from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor",
}


def compute_dataset_stats(labels_dir: Path) -> dict:
    class_counts: Counter[int] = Counter()
    box_areas: list[float] = []

    label_paths = sorted(labels_dir.glob("*.txt"))

    for label_path in label_paths:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            _, _, w, h = map(float, parts[1:])

            class_counts[class_id] += 1
            box_areas.append(w * h)

    areas = np.array(box_areas)

    stats = {
        "labels_directory": str(labels_dir),
        "num_label_files": len(label_paths),
        "num_boxes": int(len(areas)),
        "class_counts": {
            CLASS_NAMES.get(class_id, str(class_id)): int(count)
            for class_id, count in sorted(class_counts.items())
        },
        "box_area": {
            "mean": float(areas.mean()),
            "median": float(np.median(areas)),
            "p10": float(np.percentile(areas, 10)),
            "p90": float(np.percentile(areas, 90)),
        },
        "approx_object_size_at_640": {
            "median_px": float(np.sqrt(np.median(areas)) * 640),
            "p10_px": float(np.sqrt(np.percentile(areas, 10)) * 640),
            "p90_px": float(np.sqrt(np.percentile(areas, 90)) * 640),
        },
        "approx_object_size_at_960": {
            "median_px": float(np.sqrt(np.median(areas)) * 960),
            "p10_px": float(np.sqrt(np.percentile(areas, 10)) * 960),
            "p90_px": float(np.sqrt(np.percentile(areas, 90)) * 960),
        },
    }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute YOLO dataset statistics.")
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Path to YOLO labels directory, e.g. labels/train.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path.",
    )

    args = parser.parse_args()

    stats = compute_dataset_stats(args.labels_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()