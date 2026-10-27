"""RDD2022 üzerinde eğitilmiş YOLO ağırlıklarıyla yol hasarı çıkarımı."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def run(model_path: Path, source: Path, output_dir: Path, confidence: float) -> int:
    model = YOLO(str(model_path))
    output_dir = output_dir.resolve()
    results = model.predict(source=str(source), conf=confidence, save=True, project=str(output_dir), name="annotated", exist_ok=True, verbose=False)
    count = 0
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "detections.jsonl").open("w", encoding="utf-8") as stream:
        for result in results:
            boxes = result.boxes
            names = result.names
            detections = []
            if boxes is not None:
                for cls, conf, xyxy in zip(boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()):
                    detections.append({"class_id": int(cls), "class_name": names[int(cls)], "confidence": round(float(conf), 4), "xyxy": [round(float(v), 2) for v in xyxy]})
            stream.write(json.dumps({"source": str(result.path), "detections": detections}, ensure_ascii=False) + "\n")
            count += len(detections)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/yolo12s_seed0_best.pt"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/outputs/road_damage"))
    parser.add_argument("--conf", type=float, default=0.10)
    args = parser.parse_args()
    count = run(args.model, args.source, args.output, args.conf)
    print(json.dumps({"status": "ok", "detections": count, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()

