"""RDD2022 Pascal VOC anotasyonlarını YOLO klasör yapısına dönüştürür."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


CLASSES = {"D00": 0, "D10": 1, "D20": 2, "D40": 3}


def split_for(name: str, validation_ratio: float) -> str:
    value = int(hashlib.sha1(name.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "val" if value < validation_ratio else "train"


def convert(source: Path, output: Path, validation_ratio: float) -> tuple[int, int]:
    xml_files = sorted(source.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"XML anotasyonu bulunamadı: {source}")
    written = {"train": 0, "val": 0}
    for xml_path in xml_files:
        root = ET.parse(xml_path).getroot()
        filename = root.findtext("filename") or ""
        image_path = xml_path.parent.parent / "images" / filename
        if not image_path.exists():
            matches = list(source.rglob(filename))
            if not matches:
                continue
            image_path = matches[0]
        size = root.find("size")
        if size is None:
            continue
        width = float(size.findtext("width", "0"))
        height = float(size.findtext("height", "0"))
        if width <= 0 or height <= 0:
            continue
        labels: list[str] = []
        for item in root.findall("object"):
            code = item.findtext("name", "").strip()
            if code not in CLASSES:
                continue
            box = item.find("bndbox")
            if box is None:
                continue
            xmin = float(box.findtext("xmin", "0")); ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0")); ymax = float(box.findtext("ymax", "0"))
            box_w = max(0.0, xmax - xmin); box_h = max(0.0, ymax - ymin)
            if box_w == 0 or box_h == 0:
                continue
            labels.append(f"{CLASSES[code]} {(xmin + xmax) / 2 / width:.6f} {(ymin + ymax) / 2 / height:.6f} {box_w / width:.6f} {box_h / height:.6f}")
        if not labels:
            continue
        split = split_for(str(image_path.relative_to(source)), validation_ratio)
        image_out = output / "images" / split / image_path.name
        label_out = output / "labels" / split / f"{image_path.stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True); label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, image_out)
        label_out.write_text("\n".join(labels) + "\n", encoding="utf-8")
        written[split] += 1
    (output / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnc: 4\nnames: [longitudinal_crack, transverse_crack, alligator_crack, pothole]\n", encoding="utf-8")
    return written["train"], written["val"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    args = parser.parse_args()
    train, val = convert(args.source, args.output, args.validation_ratio)
    print(f"Hazırlandı: train={train}, val={val}")


if __name__ == "__main__":
    main()

