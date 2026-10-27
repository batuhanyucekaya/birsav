"""Konteyner sınıf klasörlerini train/val görüntü klasörlerine ayırır."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


EXTENSIONS = {".jpg", ".jpeg", ".png"}


def prepare(source: Path, output: Path, validation_ratio: float) -> dict[str, int]:
    counts = {"train": 0, "val": 0}
    for label_dir in sorted(p for p in source.iterdir() if p.is_dir()):
        images = sorted(p for p in label_dir.iterdir() if p.suffix.lower() in EXTENSIONS)
        if not images:
            continue
        for image in images:
            digest = int(hashlib.sha1(image.name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
            split = "val" if digest < validation_ratio else "train"
            target = output / split / label_dir.name
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target / image.name)
            counts[split] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/container"))
    parser.add_argument("--output", type=Path, default=Path("data/container_dataset"))
    args = parser.parse_args()
    print(prepare(args.source, args.output, 0.2))


if __name__ == "__main__":
    main()

