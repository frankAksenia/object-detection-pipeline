from pathlib import Path

from src.data.dataset_statistics import compute_yolo_label_stats


def test_compute_yolo_label_stats(tmp_path: Path):
    labels_dir = tmp_path / "labels" / "train"
    labels_dir.mkdir(parents=True)

    label_file = labels_dir / "sample.txt"
    label_file.write_text(
        "\n".join(
            [
                "0 0.500000 0.500000 0.100000 0.200000",
                "3 0.250000 0.250000 0.050000 0.050000",
            ]
        ),
        encoding="utf-8",
    )

    empty_file = labels_dir / "empty.txt"
    empty_file.write_text("", encoding="utf-8")

    stats = compute_yolo_label_stats(labels_dir)

    assert stats["num_label_files"] == 2
    assert stats["num_empty_label_files"] == 1
    assert stats["num_boxes"] == 2
    assert stats["class_distribution"]["pedestrian"]["count"] == 1
    assert stats["class_distribution"]["car"]["count"] == 1
    assert stats["quality_checks"]["invalid_lines"] == 0
    assert stats["quality_checks"]["invalid_coordinates"] == 0
