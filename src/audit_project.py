"""Read-only model/data audit. Reports are local; labels and weights are never modified."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(path):
    return (
        sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)
        if path.exists()
        else []
    )


def validate_label(text, nc, task="detect"):
    findings = []
    classes = Counter()
    seen = set()
    for line_num, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            values = [float(x) for x in line.split()]
            canonical = tuple(values)
            if canonical in seen:
                findings.append(
                    {
                        "line": line_num,
                        "code": "duplicate_annotation",
                        "severity": "review",
                    }
                )
            seen.add(canonical)
            if not all(math.isfinite(x) for x in values):
                raise ValueError("nonfinite")
            cls = values[0]
            if not cls.is_integer() or not 0 <= cls < nc:
                raise ValueError("class_id")
            coords = values[1:]
            if task == "segment":
                if len(coords) < 6 or len(coords) % 2:
                    raise ValueError("polygon_format")
                points = np.array(coords).reshape(-1, 2)
                if (points < 0).any() or (points > 1).any():
                    raise ValueError("polygon_bounds")
                if (
                    len(np.unique(points, axis=0)) < 3
                    or cv2.contourArea(points.astype(np.float32)) <= 1e-10
                ):
                    raise ValueError("polygon_zero_area")
            else:
                if len(coords) != 4:
                    raise ValueError("bbox_format")
                x, y, w, h = coords
                if w <= 0 or h <= 0:
                    raise ValueError("bbox_zero_area")
                if (
                    min(x - w / 2, y - h / 2) < -1e-5
                    or max(x + w / 2, y + h / 2) > 1 + 1e-5
                ):
                    raise ValueError("bbox_bounds")
                if w * h >= 0.90:
                    findings.append(
                        {
                            "line": line_num,
                            "code": "full_image_bbox",
                            "severity": "review",
                        }
                    )
            classes[int(cls)] += 1
        except (ValueError, IndexError) as exc:
            findings.append({"line": line_num, "code": str(exc), "severity": "error"})
    return findings, classes


def audit_dataset(config_path, near_distance=3):
    config_path = Path(config_path).resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = Path(cfg.get("path", config_path.parent))
    if not base.is_absolute():
        base = (config_path.parent / base).resolve()
    names = cfg.get("names", [])
    task = "segment" if "mapillary" in str(config_path).lower() else "detect"
    report = {
        "config": str(config_path),
        "root": str(base),
        "task": task,
        "names": names,
        "splits": {},
        "findings": [],
        "exact_cross_split_duplicates": [],
        "near_cross_split_candidates": [],
        "group_leakage": "unknown: source video/route/day manifest required",
    }
    hashes = defaultdict(list)
    perceptual = []
    for split in ("train", "val", "test"):
        entry = cfg.get(split)
        if entry is None:
            report["splits"][split] = {"configured": False, "images": 0}
            continue
        entries = entry if isinstance(entry, list) else [entry]
        paths = []
        for value in entries:
            folder = Path(value)
            folder = folder if folder.is_absolute() else base / folder
            if folder.suffix == ".txt" and folder.exists():
                for line in folder.read_text(encoding="utf-8").splitlines():
                    p = Path(line)
                    paths.append(p if p.is_absolute() else folder.parent / p)
            else:
                paths.extend(image_paths(folder))
        counts = Counter()
        missing = empty = 0
        for image in paths:
            frame = cv2.imread(str(image))
            if frame is None:
                report["findings"].append(
                    {
                        "image": str(image),
                        "code": "unreadable_image",
                        "severity": "error",
                    }
                )
                continue
            # Hash decoded pixels, so container/encoding differences do not hide duplicates.
            digest = hashlib.sha256(
                str(frame.shape).encode() + frame.tobytes()
            ).hexdigest()
            hashes[digest].append({"split": split, "image": str(image)})
            thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8))
            bits = (thumb[:, 1:] > thumb[:, :-1]).flatten()
            dhash = sum(int(b) << i for i, b in enumerate(bits))
            perceptual.append((split, str(image), dhash, digest))
            parts = list(image.parts)
            if "images" not in parts:
                report["findings"].append(
                    {
                        "image": str(image),
                        "code": "unknown_label_location",
                        "severity": "error",
                    }
                )
                continue
            parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
            label = Path(*parts).with_suffix(".txt")
            if not label.exists():
                missing += 1
                report["findings"].append(
                    {"image": str(image), "code": "missing_label", "severity": "error"}
                )
                continue
            content = label.read_text(encoding="utf-8-sig")
            empty += not content.strip()
            findings, classes = validate_label(content, len(names), task)
            counts.update(classes)
            report["findings"].extend(
                {**f, "label": str(label), "image": str(image)} for f in findings
            )
        report["splits"][split] = {
            "configured": True,
            "images": len(paths),
            "missing_labels": missing,
            "empty_labels": empty,
            "objects_per_class": dict(counts),
        }
        # Orphans can indicate renamed/lost images; report instead of deleting.
        label_dir = base / "labels" / split
        stems = {p.stem for p in paths}
        for label in sorted(label_dir.glob("*.txt")):
            if label.stem not in stems:
                report["findings"].append(
                    {"label": str(label), "code": "orphan_label", "severity": "error"}
                )
    report["exact_cross_split_duplicates"] = [
        v for v in hashes.values() if len({x["split"] for x in v}) > 1
    ]
    report["within_split_duplicate_groups"] = [
        v for v in hashes.values() if len(v) > 1 and len({x["split"] for x in v}) == 1
    ]
    for i, (split, path, h, digest) in enumerate(perceptual):
        for other_split, other_path, other_h, other_digest in perceptual[i + 1 :]:
            if (
                split != other_split
                and digest != other_digest
                and (h ^ other_h).bit_count() <= near_distance
            ):
                report["near_cross_split_candidates"].append(
                    {
                        "a": path,
                        "b": other_path,
                        "distance": (h ^ other_h).bit_count(),
                        "status": "manual_review_required",
                    }
                )
    report["error_count"] = sum(f["severity"] == "error" for f in report["findings"])
    report["review_count"] = sum(f["severity"] == "review" for f in report["findings"])
    return report


def model_inventory(root=ROOT):
    import torch
    from ultralytics import YOLO

    models = []
    paths = sorted(
        set(root.glob("*.pt"))
        | set((root / "models").rglob("*.pt"))
        | set((root / "runs").rglob("best.pt"))
    )
    for path in paths:
        item = {
            "path": str(path.relative_to(root)),
            "sha256": checksum(path),
            "bytes": path.stat().st_size,
            "independent_test": None,
            "provenance": "checkpoint metadata is not independently verified",
        }
        try:
            model = YOLO(str(path))
            ckpt = model.ckpt or {}
            item.update(
                task=model.task,
                names=model.names,
                architecture=type(model.model).__name__,
                yaml=getattr(model.model, "yaml", None),
                parameters=sum(p.numel() for p in model.model.parameters()),
                checkpoint_version=ckpt.get("version"),
                date=ckpt.get("date"),
                epoch=ckpt.get("epoch"),
                train_args=ckpt.get("train_args"),
                best_checkpoint_metrics=ckpt.get("train_metrics"),
            )
            run_dir = path.parent.parent
            args = run_dir / "args.yaml"
            if args.exists():
                item["run_args"] = yaml.safe_load(args.read_text(encoding="utf-8"))
            results = run_dir / "results.csv"
            if results.exists():
                with results.open(encoding="utf-8-sig", newline="") as f:
                    rows = list(csv.DictReader(f))
                item["recorded_epochs"] = len(rows)
                item["last_epoch"] = rows[-1] if rows else None
            del model
        except Exception as exc:
            item["load_error"] = f"{type(exc).__name__}: {exc}"
        models.append(item)
    for path in sorted((root / "models" / "privacy").glob("*")):
        if path.is_file():
            models.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": checksum(path),
                    "bytes": path.stat().st_size,
                    "task": "privacy_detection",
                    "training_provenance": None,
                    "independent_test": None,
                }
            )
    return models


def hardware():
    import torch
    import psutil

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "ram": psutil.virtual_memory()._asdict(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_build": torch.version.cuda,
        "gpus": [
            torch.cuda.get_device_properties(i).__repr__()
            for i in range(torch.cuda.device_count())
        ],
        "libraries": {
            n: importlib.metadata.version(n)
            for n in ("ultralytics", "numpy", "psutil", "PyYAML")
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/project_audit.json")
    )
    parser.add_argument("--skip-models", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    result = {
        "hardware": hardware(),
        "models": [] if args.skip_models else model_inventory(),
        "datasets": [],
    }
    for cfg in sorted((ROOT / "data").rglob("data.yaml")):
        print(f"Audit: {cfg.relative_to(ROOT)}", flush=True)
        result["datasets"].append(audit_dataset(cfg))
    for folder in (
        ROOT / "data" / "container_dataset",
        ROOT / "data" / "overflow_detection",
    ):
        result.setdefault("additional_data", {})[str(folder.relative_to(ROOT))] = {
            s: len(
                image_paths(
                    folder / s if "dataset" in folder.name else folder / "images" / s
                )
            )
            for s in ("train", "val", "test")
        }
    result["elapsed_seconds"] = time.perf_counter() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
