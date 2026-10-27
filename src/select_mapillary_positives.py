"""Ek Mapillary örneklerinden kaldırım/konteyner pozitiflerini seçer."""
from pathlib import Path
import json
import shutil
import cv2
import numpy as np

SRC = Path("data/mapillary_extra")
DST = Path("data/mapillary_small")
cfg = json.loads((DST / "config_v2.0.json").read_text(encoding="utf-8"))
colors = {i: tuple(reversed(x["color"])) for i, x in enumerate(cfg["labels"])}

def main():
    counts = {i: 0 for i in [24, 104, 66, 78, 79, 80, 81, 82, 83, *range(96, 104)]}
    for mask_path in (SRC / "training" / "v2.0" / "labels").glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        pixels = {i: int(np.all(mask == np.array(colors[i], dtype=np.uint8), axis=2).sum()) for i in counts}
        if any(value > 100 for value in pixels.values()):
            image = SRC / "training" / "images" / f"{mask_path.stem}.jpg"
            if not image.exists():
                continue
            shutil.copy2(image, DST / "training" / "images" / image.name)
            shutil.copy2(mask_path, DST / "training" / "v2.0" / "labels" / mask_path.name)
            for i, value in pixels.items():
                counts[i] += bool(value > 100)
    print("pozitif kare sayıları:", counts)

if __name__ == "__main__":
    main()
