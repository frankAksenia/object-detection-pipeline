from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# https://docs.ultralytics.com/datasets/detect/visdrone
# https://github.com/VisDrone/VisDrone-Dataset


VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip",
}


def extraction_complete(extracted_split_dir: Path) -> bool:
    images_dir = extracted_split_dir / "images"
    annotations_dir = extracted_split_dir / "annotations"

    if not images_dir.is_dir() or not annotations_dir.is_dir():
        return False

    return any(images_dir.glob("*.jpg")) and any(annotations_dir.glob("*.txt"))


def download_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"Already exists: {output_path}")
        return

    print(f"Downloading: {url}")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))

    with output_path.open("wb") as file, tqdm(total=total, unit="B", unit_scale=True, desc=output_path.name) as progress:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                progress.update(len(chunk))


def unzip_file(zip_path: Path, output_dir: Path, force: bool = False) -> None:
    expected_dir = output_dir / zip_path.stem

    if expected_dir.exists() and not force and extraction_complete(expected_dir):
        print(f"Already extracted: {expected_dir}")
        return

    if expected_dir.exists():
        print(f"Removing incomplete extraction: {expected_dir}")
        shutil.rmtree(expected_dir)

    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    if not extraction_complete(expected_dir):
        raise RuntimeError(f"Extraction did not produce expected images and annotations in {expected_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VisDrone DET dataset.")

    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where VisDrone raw data will be stored.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"], help="Dataset splits to download.")
    parser.add_argument("--extract", action="store_true", help="Extract zip files after downloading.")
    parser.add_argument("--force-extract", action="store_true", help="Remove and re-extract split folders even if they look complete.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_dir = args.output_dir / "zips"
    extracted_dir = args.output_dir / "extracted"

    for split in args.splits:
        url = VISDRONE_URLS[split]
        zip_path = zip_dir / Path(url).name

        download_file(url=url, output_path=zip_path)

        if args.extract:
            unzip_file(zip_path=zip_path, output_dir=extracted_dir, force=args.force_extract)

    print("VisDrone download complete.")


if __name__ == "__main__":
    main()
