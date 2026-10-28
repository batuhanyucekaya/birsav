"""Yol/kaldırım ve taşan çöp ilişkisi için YOLO detection eğitimi."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO
from src.audit_project import audit_dataset
import torch
import shutil

CLASSES = ["container", "overflowing_litter", "ground_litter"]


def validate(root: Path) -> int:
    images = list((root / "images" / "train").glob("*"))
    if not images:
        raise SystemExit(
            "Henüz görsel yok: data/overflow_detection/images/train içine görseller ekleyin."
        )
    missing = [
        p for p in images if not (root / "labels" / "train" / f"{p.stem}.txt").exists()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} görselin YOLO etiketi eksik: {missing[0]}")
    return len(images)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/overflow_detection"))
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    if shutil.disk_usage(Path.cwd()).free < 512 * 1024 * 1024:
        raise SystemExit("Eğitim için en az 512 MB boş disk alanı gerekli")
    count = validate(args.root)
    yaml = args.root / "data.yaml"
    yaml.write_text(
        f"path: {args.root.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"nc: {len(CLASSES)}\nnames: [{', '.join(CLASSES)}]\n",
        encoding="utf-8",
    )
    report = audit_dataset(yaml)
    if (
        report["error_count"]
        or report["exact_cross_split_duplicates"]
        or not report["splits"]["val"]["images"]
    ):
        raise SystemExit(
            "Eğitim engellendi: etiket, doğrulama kümesi veya sızıntı denetimi geçmedi"
        )
    print(
        f"{count} eğitim görseli doğrulandı; eğitim başlıyor. Bağımsız saha testi ayrıca gereklidir."
    )
    model = YOLO("yolo11n.pt")
    torch.set_num_threads(2)
    model.train(
        data=str(yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=2,
        workers=0,
        device=0 if torch.cuda.is_available() else "cpu",
        cache=False,
        seed=0,
        project="runs/overflow",
        name="train",
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
