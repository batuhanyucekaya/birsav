"""Mapillary Vistas semantic masklerini küçük YOLO segmentasyon setine çevirir."""

from __future__ import annotations

import random
import shutil
import json
from pathlib import Path

import cv2
import numpy as np


SRC = Path("data/mapillary_small")
OUT = Path("data/mapillary_seg")
# YOLO sınıfları: yol, kaldırım, şerit/işaret, konteyner, pano ve tabela.
TARGETS = {
    21: 0,
    24: 1,
    **{i: 2 for i in range(35, 59)},
    104: 3,
    66: 4,
    78: 4,  # object--sign--advertisement (reklam panosu)
    **{i: 5 for i in [79, 80, 81, 82, 83, *range(96, 104)]},
}
NAMES = ["road", "sidewalk", "lane_marking", "trash_can", "billboard", "sign"]
CONFIG = json.loads((SRC / "config_v2.0.json").read_text(encoding="utf-8"))
COLORS = {i: tuple(reversed(item["color"])) for i, item in enumerate(CONFIG["labels"])}


def convert_mask(mask_path: Path, label_path: Path) -> None:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return
    scale = min(1.0, 1024.0 / max(mask.shape[:2]))
    if scale < 1.0:
        mask = cv2.resize(mask, (int(mask.shape[1] * scale), int(mask.shape[0] * scale)), interpolation=cv2.INTER_NEAREST)
    height, width = mask.shape[:2]
    lines = []
    for source_id, target_id in TARGETS.items():
        binary = np.all(mask == np.array(COLORS[source_id], dtype=np.uint8), axis=2).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) < 30 or len(contour) < 3:
                continue
            epsilon = max(1.0, 0.002 * cv2.arcLength(contour, True))
            polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
            if len(polygon) < 3:
                continue
            coords = " ".join(f"{x / width:.6f} {y / height:.6f}" for x, y in polygon)
            lines.append(f"{target_id} {coords}")
    label_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    images = sorted((SRC / "training" / "images").glob("*.jpg"))
    random.Random(42).shuffle(images)
    split_at = max(1, int(len(images) * 0.8))
    for split, group in (("train", images[:split_at]), ("val", images[split_at:])):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for image in group:
            shutil.copy2(image, OUT / "images" / split / image.name)
            convert_mask(SRC / "training" / "v2.0" / "labels" / f"{image.stem}.png", OUT / "labels" / split / f"{image.stem}.txt")
    (OUT / "data.yaml").write_text(
        f"path: {OUT.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnc: {len(NAMES)}\nnames: {NAMES}\n",
        encoding="utf-8",
    )
    print(f"Hazır: train={len(images[:split_at])}, val={len(images[split_at:])}")


if __name__ == "__main__":
    main()
