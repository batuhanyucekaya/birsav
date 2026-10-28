"""Mine local FP/FN review candidates. Suggestions are never accepted as ground truth."""

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import yaml
from src.audit_project import image_paths
from src.privacy import PrivacyProcessor, mask_regions
from src.tracking import iou, BoxTracker


def match_detections(predictions, truth, threshold=0.5):
    used = set()
    tp = []
    fp = []
    for pred in sorted(predictions, key=lambda p: p["confidence"], reverse=True):
        candidates = [
            (iou(pred["box"], gt["box"]), i)
            for i, gt in enumerate(truth)
            if i not in used and gt["class_name"] == pred["class_name"]
        ]
        score, index = max(candidates, default=(0, None))
        if index is not None and score >= threshold:
            used.add(index)
            tp.append(pred)
        else:
            fp.append(pred)
    return tp, fp, [gt for i, gt in enumerate(truth) if i not in used]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--split", choices=["val", "test"], default="val")
    args = p.parse_args()
    cfg = yaml.safe_load(args.data.read_text())
    root = Path(cfg["path"])
    folder = root / cfg[args.split]
    torch.set_num_threads(2)
    model = YOLO(str(args.model))
    items = []
    for path in image_paths(folder):
        frame = cv2.imread(str(path))
        h, w = frame.shape[:2]
        prediction = model.predict(frame, imgsz=320, conf=args.conf, verbose=False)[0]
        predicted = [
            {"box": b, "class_name": int(c), "confidence": float(s)}
            for b, c, s in zip(
                prediction.boxes.xyxy.tolist(),
                prediction.boxes.cls.tolist(),
                prediction.boxes.conf.tolist(),
            )
        ]
        truth = []
        label = root / "labels" / args.split / f"{path.stem}.txt"
        for row in label.read_text().splitlines():
            values = list(map(float, row.split()))
            cls = int(values[0])
            coords = values[1:]
            if len(coords) == 4:
                x, y, bw, bh = coords
                box = [
                    (x - bw / 2) * w,
                    (y - bh / 2) * h,
                    (x + bw / 2) * w,
                    (y + bh / 2) * h,
                ]
            else:
                points = np.array(coords).reshape(-1, 2)
                box = [
                    points[:, 0].min() * w,
                    points[:, 1].min() * h,
                    points[:, 0].max() * w,
                    points[:, 1].max() * h,
                ]
            truth.append({"class_name": cls, "box": box})
        tp, fp, fn = match_detections(predicted, truth)
        items.append(
            {
                "image": str(path),
                "tp": len(tp),
                "fp": fp,
                "fn": fn,
                "status": "manual_review_required",
                "automatic_ground_truth": False,
            }
        )
    items.sort(key=lambda x: len(x["fp"]) + len(x["fn"]), reverse=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "candidates.json").write_text(
        json.dumps(
            {
                "purpose": "fixed_threshold_box_error_review_NOT_AP",
                "confidence_threshold": args.conf,
                "split": args.split,
                "items": items,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    privacy = PrivacyProcessor(Path(__file__).resolve().parent.parent)
    worst = items[:12]
    canvas = np.full((max(1, (len(worst) + 2) // 3) * 280, 960, 3), 255, np.uint8)
    for i, item in enumerate(worst):
        frame = cv2.imread(item["image"])
        privacy.tracker = BoxTracker(0.2, alpha=1, max_missing=10)
        redacted, regions = privacy.process(frame)
        for kind, color in [("fp", (0, 0, 255)), ("fn", (0, 180, 255))]:
            for box in item[kind]:
                a, b, c, d = map(int, box["box"])
                cv2.rectangle(redacted, (a, b), (c, d), color, 3)
        redacted = mask_regions(redacted, regions)
        h, w = redacted.shape[:2]
        scale = min(320 / w, 240 / h)
        small = cv2.resize(redacted, (round(w * scale), round(h * scale)))
        x = (i % 3) * 320
        y = (i // 3) * 280
        canvas[y : y + small.shape[0], x : x + small.shape[1]] = small
        cv2.putText(
            canvas,
            f'#{i+1} FP={len(item["fp"])} FN={len(item["fn"])}',
            (x + 4, y + 260),
            0,
            0.5,
            (0, 0, 0),
            1,
        )
    cv2.imwrite(str(args.output / "worst_cases.jpg"), canvas)
    print(
        f"{len(items)} images reviewed; suggestions saved locally, not used for training"
    )


if __name__ == "__main__":
    main()
