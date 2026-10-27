"""Konteyner detection veri setini kontrol eder ve YOLO data.yaml üretir."""

from __future__ import annotations

import argparse
from pathlib import Path


CLASSES = ["empty", "full", "overflowing"]


def prepare(root: Path) -> None:
    missing = []
    for split in ("train", "val"):
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        for image in sorted(image_dir.glob("*")) if image_dir.exists() else []:
            label = label_dir / f"{image.stem}.txt"
            if not label.exists():
                missing.append(str(label))
    root.mkdir(parents=True, exist_ok=True)
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 3\nnames: [empty, full, overflowing]\n", encoding="utf-8")
    if missing:
        raise SystemExit(f"Eksik YOLO etiketleri: {len(missing)}. Örnek: {missing[0]}")
    print("Detection veri seti hazır: data.yaml ve tüm etiketler mevcut.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/container_detection"))
    prepare(parser.parse_args().root)

