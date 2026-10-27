"""Yol/kaldırım ve taşan çöp ilişkisi için YOLO detection eğitimi."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


CLASSES = ["container", "overflowing_litter", "ground_litter"]


def validate(root: Path) -> int:
    images = list((root / "images" / "train").glob("*"))
    if not images:
        raise SystemExit("Henüz görsel yok: data/overflow_detection/images/train içine görseller ekleyin.")
    missing = [p for p in images if not (root / "labels" / "train" / f"{p.stem}.txt").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} görselin YOLO etiketi eksik: {missing[0]}")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/overflow_detection"))
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    count = validate(args.root)
    yaml = args.root / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n",
        encoding="utf-8",
    )
    print(f"{count} eğitim görseli doğrulandı; eğitim başlıyor.")
    model = YOLO("yolo11n.pt")
    model.train(data=str(yaml), epochs=args.epochs, imgsz=640, project="runs/overflow", name="train", exist_ok=True)


if __name__ == "__main__":
    main()
