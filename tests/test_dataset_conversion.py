from pathlib import Path

from PIL import Image

from src.data.convert_visdrone_to_yolo import convert_annotation


def test_convert_visdrone_annotation_to_yolo(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    annotation_path = tmp_path / "sample.txt"
    output_label_path = tmp_path / "sample_yolo.txt"

    image = Image.new("RGB", (1000, 500), color="white")
    image.save(image_path)

    # VisDrone format:
    # bbox_left,bbox_top,bbox_width,bbox_height,
    # score,object_category,truncation,occlusion
    # Raw class 4 = car, YOLO class should be 3.
    annotation_path.write_text(
        "100,50,200,100,1,4,0,0\n",
        encoding="utf-8",
    )

    convert_annotation(
        annotation_path=annotation_path,
        image_path=image_path,
        output_label_path=output_label_path,
    )

    lines = output_label_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 1

    parts = lines[0].split()
    assert int(parts[0]) == 3

    x_center = float(parts[1])
    y_center = float(parts[2])
    width = float(parts[3])
    height = float(parts[4])

    assert x_center == 0.2
    assert y_center == 0.2
    assert width == 0.2
    assert height == 0.2


def test_convert_visdrone_skips_ignored_regions(tmp_path: Path):
    image_path = tmp_path / "sample.jpg"
    annotation_path = tmp_path / "sample.txt"
    output_label_path = tmp_path / "sample_yolo.txt"

    image = Image.new("RGB", (1000, 500), color="white")
    image.save(image_path)

    # score 0 should be skipped.
    annotation_path.write_text(
        "100,50,200,100,0,4,0,0\n",
        encoding="utf-8",
    )

    convert_annotation(
        annotation_path=annotation_path,
        image_path=image_path,
        output_label_path=output_label_path,
    )

    assert output_label_path.read_text(encoding="utf-8") == ""
