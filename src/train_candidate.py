"""Controlled local fine-tuning. Writes a candidate; never activates it."""

import argparse
import json
from pathlib import Path
import time
import shutil
import torch
from ultralytics import YOLO
from src.audit_project import checksum, validate_label
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    if shutil.disk_usage(Path.cwd()).free < 512 * 1024 * 1024:
        raise ValueError(
            "At least 512 MB free disk space required before training; existing checkpoints are preserved"
        )
    cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    root = Path(cfg["path"])
    root = root if root.is_absolute() else args.data.resolve().parent / root
    if not args.model.is_file():
        raise ValueError("Local model required; download disabled")
    for label in (root / "labels").rglob("*.txt"):
        findings, _ = validate_label(label.read_text(), len(cfg["names"]), "segment")
        if findings:
            raise ValueError(f"Invalid/review-required label: {label}")
    torch.set_num_threads(2)
    model = YOLO(str(args.model))
    start = time.perf_counter()
    result = model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=320,
        batch=2,
        workers=0,
        device="cpu",
        cache=False,
        optimizer="AdamW",
        lr0=0.0002,
        nbs=16,
        warmup_epochs=0,
        mosaic=0,
        fliplr=0,
        mixup=0,
        copy_paste=0,
        erasing=0,
        seed=0,
        amp=False,
        deterministic=True,
        project="runs/candidates",
        name=args.name,
        exist_ok=False,
        plots=False,
    )
    report = {
        "name": args.name,
        "purpose": "development_candidate_NOT_independent_field_test",
        "promoted": False,
        "starting_model": str(args.model),
        "starting_sha256": checksum(args.model),
        "data": str(args.data),
        "train_config": vars(model.trainer.args),
        "metrics": result.results_dict,
        "elapsed_seconds": time.perf_counter() - start,
        "best": str(model.trainer.best),
        "best_sha256": checksum(model.trainer.best),
    }
    Path(f"reports/{args.name}_training.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
