"""Konteyner detection veri setini kontrol eder ve YOLO data.yaml üretir."""

from __future__ import annotations

import argparse
from pathlib import Path
from src.audit_project import audit_dataset

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
    if missing:
        raise SystemExit(f"Eksik YOLO etiketleri: {len(missing)}. Örnek: {missing[0]}")
    root.mkdir(parents=True, exist_ok=True)
    config = root / "data.yaml"
    if not config.exists():
        config.write_text(
            f"path: {root.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnc: 3\nnames: [empty, full, overflowing]\n",
            encoding="utf-8",
        )
    report = audit_dataset(config)
    if any(not report["splits"][s]["images"] for s in ("train", "val")):
        raise SystemExit("Eğitim ve doğrulama kümeleri boş olamaz")
    if (
        report["error_count"]
        or report["review_count"]
        or report["exact_cross_split_duplicates"]
    ):
        raise SystemExit(
            "Etiket/veri denetimi geçmedi: tam-görsel kutusu, veri sızıntısı veya bozuk etiket; eğitim engellendi."
        )
    print("Detection veri seti hazır: data.yaml ve tüm etiketler mevcut.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/container_detection"))
    prepare(parser.parse_args().root)
