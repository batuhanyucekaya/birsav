"""Büyük Mapillary arşivinden hedef sınıf içeren sınırlı bir alt küme çıkarır."""
from pathlib import Path
import json
import zipfile
import cv2
import numpy as np

ZIP = iter(sorted(Path(r"C:/Users/batuh/Downloads").glob("*.zip"), key=lambda p: p.stat().st_size, reverse=True)).__next__()
OUT = Path("data/mapillary_archive_extra")
MAX_SCAN = 5000
MAX_SELECTED = 1000
TARGET_IDS = [24, 104, 66, 78, 79, 80, 81, 82, 83, *range(96, 104)]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "training" / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "training" / "v2.0" / "labels").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        config = json.loads(z.read("config_v2.0.json"))
        colors = {i: tuple(reversed(x["color"])) for i, x in enumerate(config["labels"])}
        names = [x for x in z.namelist() if x.startswith("training/v2.0/labels/") and x.endswith(".png")][:MAX_SCAN]
        selected = 0
        for label_name in names:
            if selected >= MAX_SELECTED:
                break
            mask = cv2.imdecode(np.frombuffer(z.read(label_name), np.uint8), cv2.IMREAD_UNCHANGED)
            if mask is None:
                continue
            hit = False
            for idx in TARGET_IDS:
                pixels = np.all(mask == np.array(colors[idx], dtype=np.uint8), axis=2).sum()
                if pixels > 100:
                    hit = True
                    break
            if not hit:
                continue
            stem = Path(label_name).stem
            image_name = f"training/images/{stem}.jpg"
            if image_name not in z.namelist():
                continue
            (OUT / "training" / "images" / f"{stem}.jpg").write_bytes(z.read(image_name))
            (OUT / "training" / "v2.0" / "labels" / f"{stem}.png").write_bytes(z.read(label_name))
            selected += 1
        (OUT / "config_v2.0.json").write_text(json.dumps(config), encoding="utf-8")
    print(f"Arşivden seçilen ek hedef görsel: {selected}")

if __name__ == "__main__":
    main()
