"""Surface-mask evidence for litter candidates; never a definitive violation."""

import cv2
import numpy as np


def litter_context(box, scene_result, image_shape, exclusions=()):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = box
    # Approximate support patch near the object's bottom, not its entire rectangle.
    xa, xb = max(0, int(x1)), min(w, int(np.ceil(x2)))
    ya, yb = max(0, int(y2 - max(2, (y2 - y1) * 0.15))), min(h, int(np.ceil(y2 + 2)))
    if xa >= xb or ya >= yb or scene_result is None or scene_result.masks is None:
        return {
            "status": "unverified",
            "surface": None,
            "support_coverage": None,
            "review_required": True,
        }
    support = {}
    for cls, mask in zip(
        scene_result.boxes.cls.tolist(), scene_result.masks.data.cpu().numpy()
    ):
        name = scene_result.names[int(cls)]
        if name not in {"road", "sidewalk", "trash_can"}:
            continue
        # Call predict(retina_masks=True) so masks align with the original image.
        if mask.shape != (h, w):
            return {
                "status": "unverified",
                "surface": None,
                "support_coverage": None,
                "review_required": True,
                "reason": "mask_shape_mismatch",
            }
        local = mask[ya:yb, xa:xb] > 0.5
        support[name] = np.logical_or(support.get(name, np.zeros_like(local)), local)
    coverage = {name: float(mask.mean()) for name, mask in support.items()}
    surface = max(
        (k for k in coverage if k in {"road", "sidewalk"}),
        key=lambda k: coverage[k],
        default=None,
    )
    foot = ((x1 + x2) / 2, y2)
    occluded = any(a <= foot[0] <= c and b <= foot[1] <= d for a, b, c, d in exclusions)
    grounded = bool(
        surface
        and coverage[surface] >= 0.6
        and coverage.get("trash_can", 0) < 0.2
        and not occluded
    )
    return {
        "status": "surface_supported_candidate" if grounded else "unverified",
        "surface": surface,
        "support_coverage": coverage.get(surface) if surface else None,
        "occluder_overlap": occluded,
        "review_required": True,
        "note": "Uncalibrated heuristic; mask support does not prove ground litter",
    }
