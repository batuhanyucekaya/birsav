"""Render visual bbox drafts for review; never enables or trains a model."""

import json
from pathlib import Path
import cv2
import numpy as np
from src.repair_dataset import link_or_copy
from src.privacy import PrivacyProcessor, mask_regions
from src.tracking import BoxTracker


def main():
    root = Path(__file__).resolve().parent.parent
    config = json.loads(
        (root / "config/container_localization_draft.json").read_text(encoding="utf-8")
    )
    target = root / "data/container_localization_draft_v1"
    if target.exists() and not (target / "annotation_review.json").exists():
        raise ValueError("Existing directory is not a draft owned by this tool")
    privacy = PrivacyProcessor(root)
    canvas = np.full((4 * 260, 3 * 360, 3), 255, np.uint8)
    for i, (relative, boxes) in enumerate(config["boxes_xyxy_normalized"].items()):
        source = root / "data/container_detection/images" / relative
        if not (target / "images" / relative).exists():
            link_or_copy(source, target / "images" / relative)
        if isinstance(boxes[0], (int, float)):
            boxes = [boxes]
        label = target / "labels" / Path(relative).with_suffix(".txt")
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text(
            "".join(
                f"0 {(x1+x2)/2:.6f} {(y1+y2)/2:.6f} {x2-x1:.6f} {y2-y1:.6f}\n"
                for x1, y1, x2, y2 in boxes
            )
        )
        image = cv2.imread(str(source))
        h, w = image.shape[:2]
        privacy.tracker = BoxTracker(0.2, alpha=1, max_missing=10)
        image, regions = privacy.process(image)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(
                image,
                (round(x1 * w), round(y1 * h)),
                (round(x2 * w), round(y2 * h)),
                (0, 0, 255),
                3,
            )
        image = mask_regions(image, regions)
        scale = min(350 / w, 220 / h)
        image = cv2.resize(image, (round(w * scale), round(h * scale)))
        x = i % 3 * 360
        y = i // 3 * 260
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        cv2.putText(canvas, relative, (x + 4, y + 240), 0, 0.4, (0, 0, 0), 1)
    target.mkdir(parents=True, exist_ok=True)
    (target / "annotation_review.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    # No data.yaml: unreviewed annotations must not silently become a training dataset.
    out = root / "tmp/reliability-baseline/container_bbox_draft_review.jpg"
    cv2.imwrite(str(out), canvas)
    print(out)


if __name__ == "__main__":
    main()
