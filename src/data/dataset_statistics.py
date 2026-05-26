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


def safe_percentile(values: np.ndarray, percentile: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(values, percentile))


def compute_yolo_label_stats(labels_dir: Path) -> dict:
    class_counts: Counter[int] = Counter()
    boxes_per_image: list[int] = []
    box_areas: list[float] = []
    box_widths: list[float] = []
    box_heights: list[float] = []
    aspect_ratios: list[float] = []

    empty_label_files = 0
    invalid_lines = 0
    invalid_class_ids = 0
    invalid_coordinates = 0

    label_paths = sorted(labels_dir.glob("*.txt"))

    for label_path in label_paths:
        lines = label_path.read_text(encoding="utf-8").splitlines()
        valid_boxes_in_file = 0

        if not lines:
            empty_label_files += 1

        for line in lines:
            parts = line.split()

            if len(parts) != 5:
                invalid_lines += 1
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
            except ValueError:
                invalid_lines += 1
                continue

            if class_id not in CLASS_NAMES:
                invalid_class_ids += 1
                continue

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                invalid_coordinates += 1
                continue

            class_counts[class_id] += 1
            valid_boxes_in_file += 1

            box_widths.append(width)
            box_heights.append(height)
            box_areas.append(width * height)
            aspect_ratios.append(width / height)

        boxes_per_image.append(valid_boxes_in_file)

    areas = np.array(box_areas, dtype=float)
    widths = np.array(box_widths, dtype=float)
    heights = np.array(box_heights, dtype=float)
    ratios = np.array(aspect_ratios, dtype=float)
    boxes_per_image_arr = np.array(boxes_per_image, dtype=float)

    total_boxes = int(len(areas))
    max_class_count = max(class_counts.values()) if class_counts else 0
    min_class_count = min(class_counts.values()) if class_counts else 0

    class_distribution = {}
    for class_id, class_name in CLASS_NAMES.items():
        count = int(class_counts.get(class_id, 0))
        percentage = float(count / total_boxes * 100) if total_boxes else 0.0
        class_distribution[class_name] = {
            "count": count,
            "percentage": percentage,
        }

    object_size_buckets = {
        "tiny_area_lt_0.0001": int(np.sum(areas < 0.0001)) if total_boxes else 0,
        "small_area_0.0001_to_0.001": (
            int(np.sum((areas >= 0.0001) & (areas < 0.001))) if total_boxes else 0
        ),
        "medium_area_0.001_to_0.01": (
            int(np.sum((areas >= 0.001) & (areas < 0.01))) if total_boxes else 0
        ),
        "large_area_gte_0.01": int(np.sum(areas >= 0.01)) if total_boxes else 0,
    }

    object_size_bucket_percentages = {
        key: float(value / total_boxes * 100) if total_boxes else 0.0
        for key, value in object_size_buckets.items()
    }

    stats = {
        "labels_dir": str(labels_dir),
        "num_label_files": len(label_paths),
        "num_empty_label_files": empty_label_files,
        "num_boxes": total_boxes,
        "boxes_per_image": {
            "mean": (
                float(boxes_per_image_arr.mean()) if len(boxes_per_image_arr) else 0.0
            ),
            "median": (
                float(np.median(boxes_per_image_arr))
                if len(boxes_per_image_arr)
                else 0.0
            ),
            "p90": safe_percentile(boxes_per_image_arr, 90),
            "max": int(boxes_per_image_arr.max()) if len(boxes_per_image_arr) else 0,
        },
        "class_distribution": class_distribution,
        "class_imbalance": {
            "max_class_count": int(max_class_count),
            "min_class_count": int(min_class_count),
            "max_to_min_ratio": (
                float(max_class_count / min_class_count) if min_class_count else None
            ),
        },
        "box_area": {
            "mean": float(areas.mean()) if total_boxes else 0.0,
            "median": float(np.median(areas)) if total_boxes else 0.0,
            "p10": safe_percentile(areas, 10),
            "p25": safe_percentile(areas, 25),
            "p75": safe_percentile(areas, 75),
            "p90": safe_percentile(areas, 90),
        },
        "box_width": {
            "mean": float(widths.mean()) if total_boxes else 0.0,
            "median": float(np.median(widths)) if total_boxes else 0.0,
            "p10": safe_percentile(widths, 10),
            "p90": safe_percentile(widths, 90),
        },
        "box_height": {
            "mean": float(heights.mean()) if total_boxes else 0.0,
            "median": float(np.median(heights)) if total_boxes else 0.0,
            "p10": safe_percentile(heights, 10),
            "p90": safe_percentile(heights, 90),
        },
        "aspect_ratio_width_over_height": {
            "mean": float(ratios.mean()) if total_boxes else 0.0,
            "median": float(np.median(ratios)) if total_boxes else 0.0,
            "p10": safe_percentile(ratios, 10),
            "p90": safe_percentile(ratios, 90),
        },
        "object_size_buckets": object_size_buckets,
        "object_size_bucket_percentages": object_size_bucket_percentages,
        "approx_object_size_pixels": {
            "imgsz_640": {
                "p10": (
                    float(np.sqrt(safe_percentile(areas, 10)) * 640)
                    if total_boxes
                    else 0.0
                ),
                "median": (
                    float(np.sqrt(np.median(areas)) * 640) if total_boxes else 0.0
                ),
                "p90": (
                    float(np.sqrt(safe_percentile(areas, 90)) * 640)
                    if total_boxes
                    else 0.0
                ),
            },
            "imgsz_960": {
                "p10": (
                    float(np.sqrt(safe_percentile(areas, 10)) * 960)
                    if total_boxes
                    else 0.0
                ),
                "median": (
                    float(np.sqrt(np.median(areas)) * 960) if total_boxes else 0.0
                ),
                "p90": (
                    float(np.sqrt(safe_percentile(areas, 90)) * 960)
                    if total_boxes
                    else 0.0
                ),
            },
            "imgsz_1280": {
                "p10": (
                    float(np.sqrt(safe_percentile(areas, 10)) * 1280)
                    if total_boxes
                    else 0.0
                ),
                "median": (
                    float(np.sqrt(np.median(areas)) * 1280) if total_boxes else 0.0
                ),
                "p90": (
                    float(np.sqrt(safe_percentile(areas, 90)) * 1280)
                    if total_boxes
                    else 0.0
                ),
            },
        },
        "quality_checks": {
            "invalid_lines": invalid_lines,
            "invalid_class_ids": invalid_class_ids,
            "invalid_coordinates": invalid_coordinates,
        },
    }

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute YOLO label statistics.")
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

    stats = compute_yolo_label_stats(args.labels_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
