"""Tek konteyneri kadraja yakın gösteren klasörlerden ilk detection veri setini üretir."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path


def main() -> None:
    source = Path("data/container")
    target = Path("data/container_detection")
    class_ids = {"empty": 0, "full": 1, "overflowing": 2}
    for label, class_id in class_ids.items():
        images = sorted((source / label).glob("*"))
        for index, image in enumerate(images):
            if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            digest = int(hashlib.sha1(image.name.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
            split = "val" if index == 0 or digest < 0.2 else "train"
            image_out = target / "images" / split / image.name
            label_out = target / "labels" / split / f"{image.stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_out)
            # İlk görseller tek konteyneri kadraja yakın aldığı için tam görüntü kutusu kullanılır.
            label_out.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
    (target / "data.yaml").write_text(f"path: {target.resolve().as_posix()}\ntrain: images/train\nval: images/val\nnc: 3\nnames: [empty, full, overflowing]\n", encoding="utf-8")
    print("İlk detection veri seti hazırlandı. Çoklu konteyner için gerçek kutu etiketleri sonradan eklenmelidir.")


if __name__ == "__main__":
    main()

