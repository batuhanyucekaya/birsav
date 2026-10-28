"""Explicit, local evaluation; never auto-download a dataset or promote a model."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from ultralytics import YOLO
import yaml

from src.audit_project import checksum, image_paths


def evaluate(
    model_path, data, split="val", imgsz=320, output=Path("reports/evaluation.json")
):
    model_path, data, output = (
        Path(model_path).resolve(),
        Path(data).resolve(),
        Path(output),
    )
    if not model_path.is_file() or not data.exists():
        raise ValueError("Explicit local model and dataset are required")
    torch.set_num_threads(2)
    model = YOLO(str(model_path))
    start = time.perf_counter()
    report = {
        "model": str(model_path),
        "sha256": checksum(model_path),
        "data": str(data),
        "split": split,
        "purpose": (
            "test_split_independence_requires_provenance"
            if split == "test"
            else "development_validation_not_field_test"
        ),
        "imgsz": imgsz,
        "torch": torch.__version__,
        "task": model.task,
    }
    if model.task == "classify":
        paths = image_paths(data / split)
        names = model.names
        if not paths:
            raise ValueError("Requested split is empty; no fallback to another split")
        matrix = np.zeros((len(names), len(names)), dtype=int)
        probabilities = []
        for path in paths:
            true = next((i for i, n in names.items() if n == path.parent.name), None)
            if true is None:
                raise ValueError(f"Unknown class directory: {path.parent.name}")
            prediction = model.predict(
                str(path), imgsz=imgsz, device="cpu", verbose=False
            )[0]
            matrix[true, prediction.probs.top1] += 1
            probabilities.append(
                {
                    "image": str(path),
                    "true": true,
                    "predicted": int(prediction.probs.top1),
                    "confidence": float(prediction.probs.top1conf),
                }
            )
        tp = matrix.diagonal()
        precision = np.divide(
            tp, matrix.sum(0), out=np.zeros(len(names)), where=matrix.sum(0) != 0
        )
        recall = np.divide(
            tp, matrix.sum(1), out=np.zeros(len(names)), where=matrix.sum(1) != 0
        )
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros(len(names)),
            where=(precision + recall) != 0,
        )
        report.update(
            images=len(paths),
            names=names,
            accuracy=float(tp.sum() / matrix.sum()),
            balanced_accuracy=float(recall[matrix.sum(1) > 0].mean()),
            macro_f1=float(f1.mean()),
            confusion_matrix=matrix.tolist(),
            precision=precision.tolist(),
            recall=recall.tolist(),
            predictions=probabilities,
        )
    else:
        cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
        if not cfg.get(split):
            raise ValueError("Requested split not configured; no fallback permitted")
        root = Path(cfg.get("path", data.parent))
        root = root if root.is_absolute() else data.parent / root
        entries = cfg[split] if isinstance(cfg[split], list) else [cfg[split]]
        paths = []
        for entry in entries:
            folder = Path(entry)
            folder = folder if folder.is_absolute() else root / folder
            if not folder.is_dir():
                raise ValueError(
                    f"Only existing local image directories accepted: {folder}"
                )
            paths.extend(image_paths(folder))
        if not paths:
            raise ValueError("Requested split is empty")
        report["images"] = len(paths)
        metrics = model.val(
            data=str(data),
            split=split,
            imgsz=imgsz,
            batch=1,
            workers=0,
            device="cpu",
            plots=False,
            save_json=False,
            conf=0.001,
            project=str(output.parent / "validation_runs"),
            name=model_path.parent.parent.name,
            verbose=False,
        )
        report["metrics"] = {k: float(v) for k, v in metrics.results_dict.items()}
        report["per_class"] = metrics.summary()
        report["per_class_ap"] = []
        for kind in ("box", "seg"):
            measured = getattr(metrics, kind, None)
            if measured is None:
                continue
            for offset, class_id in enumerate(measured.ap_class_index):
                report["per_class_ap"].append(
                    {
                        "kind": kind,
                        "class_id": int(class_id),
                        "class_name": model.names[int(class_id)],
                        "ap50": float(measured.ap50[offset]),
                        "ap50_95": float(measured.ap[offset]),
                    }
                )
        matrix = getattr(getattr(metrics, "confusion_matrix", None), "matrix", None)
        if matrix is not None:
            report["confusion_matrix"] = matrix.tolist()
        per_image = getattr(getattr(metrics, "box", None), "image_metrics", None)
        if per_image is not None:
            report["per_image"] = per_image
        report["speed_ms_per_image"] = metrics.speed
    fingerprint = hashlib.sha256()
    for path in sorted(paths):
        fingerprint.update(path.name.encode())
        fingerprint.update(checksum(path).encode())
        if model.task != "classify":
            label = root / "labels" / split / f"{path.stem}.txt"
            if label.exists():
                fingerprint.update(checksum(label).encode())
        else:
            fingerprint.update(path.parent.name.encode())
    report["data_fingerprint"] = fingerprint.hexdigest()
    report["elapsed_seconds"] = time.perf_counter() - start
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "test", "train"], default="val")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.model, args.data, args.split, args.imgsz, args.output)


if __name__ == "__main__":
    main()
