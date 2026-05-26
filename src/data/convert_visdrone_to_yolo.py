from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

"""
Convert VisDrone2019-DET annotations to YOLO format.

The VisDrone DET annotation format is:
    [0] <bbox_left> x-coordinate
    [1] <bbox_top> y-coordinate
    [2] <bbox_width> 
    [3] <bbox_height>
    [4] <score> confidence flag where 1 means the entry
      is an active ground-truth annotation, and 0 means it should be ignored
    [5] <object_category> class id map
    [6] <truncation>
    [7] <occlusion>

This converter follows the same core approach used by Ultralytics'
VisDrone dataset conversion script:
    - skip ignored/invalid regions where score == 0
    - convert raw VisDrone class IDs 1-10 to YOLO class IDs 0-9
    - convert absolute xywh boxes to normalized YOLO xywh format
    - write labels into labels/{split}
    - organize images into images/{split}

Reference:
    Ultralytics VisDrone YOLO conversion:
    https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/VisDrone.yaml
"""


VISDRONE_CLASSES = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}


SPLIT_FOLDERS = {
    "train": "VisDrone2019-DET-train",
    "val": "VisDrone2019-DET-val",
    "test": "VisDrone2019-DET-test-dev",
}


def convert_annotation(
    annotation_path: Path, image_path: Path, output_label_path: Path
) -> None:
    image = Image.open(image_path)
    image_width, image_height = image.size

    lines: list[str] = []

    with annotation_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            raw_line_parts = raw_line.strip().split(",")

            x, y, w, h = map(float, raw_line_parts[:4])
            score = int(raw_line_parts[4])
            class_id = int(raw_line_parts[5])

            # class_id 0 = ignored region.
            if class_id not in VISDRONE_CLASSES:
                continue

            # some annotations may be marked as invalid.
            if score == 0:
                continue

            # convert to YOLO format
            x_center = (x + w / 2) / image_width
            y_center = (y + h / 2) / image_height
            width = w / image_width
            height = h / image_height

            # YOLO class IDs must be zero-based.
            yolo_class_id = class_id - 1

            lines.append(
                f"{yolo_class_id} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}"
            )

    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    output_label_path.write_text("\n".join(lines), encoding="utf-8")


def convert_split(raw_root: Path, output_root: Path, split: str) -> None:
    source_folder = raw_root / "extracted" / SPLIT_FOLDERS[split]
    source_images_dir = source_folder / "images"
    source_annotations_dir = source_folder / "annotations"

    output_images_dir = output_root / "images" / split
    output_labels_dir = output_root / "labels" / split

    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(source_images_dir.glob("*.jpg"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {source_images_dir}")

    for image_path in tqdm(image_paths, desc=f"Converting {split}"):
        annotation_path = source_annotations_dir / f"{image_path.stem}.txt"
        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / f"{image_path.stem}.txt"

        if not output_image_path.exists():
            shutil.copy2(image_path, output_image_path)

        if annotation_path.exists():
            convert_annotation(
                annotation_path=annotation_path,
                image_path=image_path,
                output_label_path=output_label_path,
            )
        else:
            output_label_path.write_text("", encoding="utf-8")


def write_data_yaml(output_root: Path) -> None:
    names = [VISDRONE_CLASSES[i] for i in sorted(VISDRONE_CLASSES)]

    yaml_text = f"""path: {output_root}
train: images/train
val: images/val
test: images/test

names:
"""

    for idx, name in enumerate(names):
        yaml_text += f"  {idx}: {name}\n"

    yaml_path = output_root / "visdrone.yaml"

    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print(f"Wrote dataset YAML: {yaml_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VisDrone DET to YOLO format.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Path to raw VisDrone directory containing extracted files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory for YOLO-formatted dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
    )

    args = parser.parse_args()

    print(f"Raw root: {args.raw_root.resolve()}")
    print(f"Output root: {args.output_root.resolve()}")

    dataset_yaml_path = args.yaml_path or args.output_root / "visdrone.yaml"
    print(f"Dataset YAML: {dataset_yaml_path.resolve()}")

    for split in args.splits:
        convert_split(
            raw_root=args.raw_root,
            output_root=args.output_root,
            split=split,
        )

    write_data_yaml(args.output_root)

    print("VisDrone YOLO conversion complete.")


if __name__ == "__main__":
    main()
