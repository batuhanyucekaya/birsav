"""One-image pipeline diagnostic. Train=val deliberately; NOT generalization evidence."""

import argparse
import json
from pathlib import Path
import shutil
import time
import torch
from ultralytics import YOLO
import yaml
from src.audit_project import checksum


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--nbs", type=int, default=1)
    parser.add_argument("--freeze-bn", action="store_true")
    parser.add_argument("--lr", type=float, default=0.003)
    args = parser.parse_args()
    if shutil.disk_usage(Path.cwd()).free < 256 * 1024 * 1024:
        raise ValueError(
            "At least 256 MB free space required for a diagnostic training run"
        )
    torch.set_num_threads(2)
    source = Path("data/ceymo_small")
    root = Path("tmp/ceymo_overfit_diagnostic").resolve()
    root.mkdir(parents=True, exist_ok=True)
    for directory in ("images/train", "labels/train"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    image = source / "images/train/1.jpg"
    label = source / "labels/train/1.txt"
    shutil.copy2(image, root / "images/train/1.jpg")
    shutil.copy2(label, root / "labels/train/1.txt")
    cfg = yaml.safe_load((source / "data.yaml").read_text())
    cfg.update(path=str(root), train="images/train", val="images/train")
    (root / "data.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    report = {
        "purpose": "pipeline_overfit_diagnostic_NOT_independent_test",
        "image_sha256": checksum(image),
        "epochs": args.epochs,
        "hardware": "cpu",
        "imgsz": 640,
        "seed": 0,
        "augmentation": "disabled",
        "pretrained": True,
        "nbs": args.nbs,
        "freeze_bn": args.freeze_bn,
        "lr": args.lr,
    }
    start = time.perf_counter()
    model = YOLO("yolo11n.pt")
    if args.freeze_bn:

        def freeze_bn(trainer):
            for module in trainer.model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()

        model.add_callback("on_train_batch_start", freeze_bn)
    metrics = model.train(
        data=str(root / "data.yaml"),
        epochs=args.epochs,
        imgsz=640,
        batch=1,
        workers=0,
        device="cpu",
        project="runs/diagnostics",
        name=f"ceymo_overfit_{args.epochs}_nbs{args.nbs}_bn{args.freeze_bn}_lr{args.lr}",
        exist_ok=False,
        optimizer="AdamW",
        lr0=args.lr,
        nbs=args.nbs,
        warmup_epochs=0,
        pretrained=True,
        mosaic=0,
        mixup=0,
        copy_paste=0,
        fliplr=0,
        flipud=0,
        translate=0,
        scale=0,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        erasing=0,
        plots=False,
        seed=0,
        deterministic=True,
        amp=False,
        cache=False,
        patience=0,
    )
    report.update(
        elapsed_seconds=time.perf_counter() - start,
        metrics=metrics.results_dict,
        best=str(model.trainer.best),
    )
    Path(
        f"reports/ceymo_overfit_{args.epochs}_nbs{args.nbs}_bn{args.freeze_bn}_lr{args.lr}.json"
    ).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
