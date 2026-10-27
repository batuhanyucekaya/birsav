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
    count_sidewalk = count_trash = 0
    for mask_path in (SRC / "training" / "v2.0" / "labels").glob("*.png"):
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        sidewalk = int(np.all(mask == np.array(colors[24], dtype=np.uint8), axis=2).sum())
        trash = int(np.all(mask == np.array(colors[104], dtype=np.uint8), axis=2).sum())
        if sidewalk or trash:
            image = SRC / "training" / "images" / f"{mask_path.stem}.jpg"
            if not image.exists():
                continue
            shutil.copy2(image, DST / "training" / "images" / image.name)
            shutil.copy2(mask_path, DST / "training" / "v2.0" / "labels" / mask_path.name)
            count_sidewalk += bool(sidewalk)
            count_trash += bool(trash)
    print(f"kaldırım pozitif: {count_sidewalk}, konteyner pozitif: {count_trash}")

if __name__ == "__main__":
    main()
