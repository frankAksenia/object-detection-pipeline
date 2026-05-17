from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment metrics into Markdown.")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        required=True,
        help="Directory containing metrics JSON files.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Output Markdown table path.",
    )

    args = parser.parse_args()

    metric_files = sorted(args.metrics_dir.glob("*.json"))

    rows = []

    for metric_file in metric_files:
        metrics = json.loads(metric_file.read_text(encoding="utf-8"))

        if "mAP50" not in metrics:
            continue

        rows.append(metrics)

    rows = sorted(rows, key=lambda row: row.get("mAP50_95", 0), reverse=True)

    lines = [
        "# Experiment Results",
        "",
        "| Experiment | Model | ImgSz | Epochs | Batch | mAP50-95 | mAP50 | mAP75 | Precision | Recall |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for row in rows:
        lines.append(
            "| "
            f"{row.get('experiment_name', '-')} | "
            f"{row.get('model', '-')} | "
            f"{row.get('imgsz', '-')} | "
            f"{row.get('epochs', '-')} | "
            f"{row.get('batch', '-')} | "
            f"{row.get('mAP50_95', 0):.4f} | "
            f"{row.get('mAP50', 0):.4f} | "
            f"{row.get('mAP75', 0):.4f} | "
            f"{row.get('mean_precision', 0):.4f} | "
            f"{row.get('mean_recall', 0):.4f} |"
        )

    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote: {args.output_md}")


if __name__ == "__main__":
    main()

# -----------Colab usage example-----------
# python src/training/collect_metrics.py \
#   --metrics-dir reports/metrics \
#   --output-md reports/experiments.md  