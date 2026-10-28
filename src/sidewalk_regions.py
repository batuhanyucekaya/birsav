"""Consolidate semantic sidewalk fragments before tracking and drawing.

Bridges are display/grouping hypotheses, never ground-truth support for litter.
The original model masks remain untouched for contextual reasoning.
"""

import cv2
import numpy as np


def consolidate_sidewalks(items, result, image_shape, occluders=()):
    selected = [x for x in items if x["class_name"] == "sidewalk"]
    if not selected or result is None or getattr(result, "masks", None) is None:
        return items
    h, w = image_shape[:2]
    masks = result.masks.data.cpu().numpy()
    if any(masks[x["mask_index"]].shape != (h, w) for x in selected):
        return items
    scale = min(1.0, 640 / max(h, w))
    mw, mh = max(1, round(w * scale)), max(1, round(h * scale))
    sx, sy = mw / w, mh / h

    def small(index):
        binary = (masks[index] > 0.5).astype(np.uint8)
        return cv2.resize(binary, (mw, mh), interpolation=cv2.INTER_NEAREST)

    sources = [(item, small(item["mask_index"])) for item in selected]
    union = np.zeros((mh, mw), np.uint8)
    road = np.zeros_like(union)
    for item, mask in sources:
        union |= mask
    for item in items:
        if item["class_name"] == "road" and masks[item["mask_index"]].shape == (h, w):
            road |= small(item["mask_index"])
    # Sidewalk evidence wins where the low-quality road model overlaps it.
    visible_road = (road != 0) & (union == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(union, 8)
    parts = []
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < 3:
            continue
        contours, _ = cv2.findContours(
            (labels == component).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        points = np.concatenate([c.reshape(-1, 2) for c in contours])
        if len(points) > 128:
            points = points[np.linspace(0, len(points) - 1, 128).astype(int)]
        parts.append((points, stats[component]))
    occluded = np.zeros_like(union)
    for x1, y1, x2, y2 in occluders:
        xa, ya = max(0, int(x1 * sx)), max(0, int(y1 * sy))
        xb, yb = min(mw, int(np.ceil(x2 * sx))), min(mh, int(np.ceil(y2 * sy)))
        occluded[ya:yb, xa:xb] = 1
    joined = union.copy()
    short_gap = max(2.0, min(mw, mh) * 0.015)
    occluded_gap = max(short_gap, min(mw, mh) * 0.18)
    inferred_bridges = []
    for i, (a, sa) in enumerate(parts):
        for b, sb in parts[i + 1 :]:
            ax, ay, aw, ah = map(int, sa[:4])
            bx, by, bw, bh = map(int, sb[:4])
            dx = max(0, ax - (bx + bw), bx - (ax + aw))
            dy = max(0, ay - (by + bh), by - (ay + ah))
            if np.hypot(dx, dy) > occluded_gap:
                continue
            # Do not join two long parallel strips sideways.
            if dx > 0 and max(ah, bh) > 2 * max(aw, bw):
                continue
            overlap_x = max(0, min(ax + aw, bx + bw) - max(ax, bx))
            overlap_y = max(0, min(ay + ah, by + bh) - max(ay, by))
            if dy > 0 and overlap_x < 0.35 * min(aw, bw):
                continue
            if dx > 0 and overlap_y < 0.35 * min(ah, bh):
                continue
            distances = ((a[:, None, :].astype(float) - b[None, :, :]) ** 2).sum(2)
            ia, ib = np.unravel_index(distances.argmin(), distances.shape)
            distance = float(np.sqrt(distances[ia, ib]))
            if distance > occluded_gap:
                continue
            pa, pb = a[ia], b[ib]
            samples = np.rint(
                np.linspace(pa, pb, max(2, int(np.ceil(distance)) + 1))
            ).astype(int)
            missing = union[samples[:, 1], samples[:, 0]] == 0
            if not missing.any():
                continue
            gap_points = samples[missing]
            if visible_road[gap_points[:, 1], gap_points[:, 0]].mean() > 0.15:
                continue
            supported = occluded[gap_points[:, 1], gap_points[:, 0]].mean() >= 0.6
            if distance > short_gap and not supported:
                continue
            cv2.line(joined, tuple(map(int, pa)), tuple(map(int, pb)), 1, 1)
            inferred_bridges.append((pa, pb))
    count, groups, stats, _ = cv2.connectedComponentsWithStats(joined, 8)
    output = [x for x in items if x["class_name"] != "sidewalk"]
    for component in range(1, count):
        region = groups == component
        members = [item for item, mask in sources if np.any(region & (mask != 0))]
        if not members:
            continue
        contours, _ = cv2.findContours(
            region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        polygons = []
        for contour in contours:
            if len(contour) < 3:
                continue
            polygon = (
                cv2.approxPolyDP(
                    contour, max(0.5, 0.001 * cv2.arcLength(contour, True)), True
                )
                .reshape(-1, 2)
                .astype(float)
            )
            polygon[:, 0] = np.clip(np.rint(polygon[:, 0] / sx), 0, w - 1)
            polygon[:, 1] = np.clip(np.rint(polygon[:, 1] / sy), 0, h - 1)
            polygons.append(polygon.astype(int).tolist())
        if not polygons:
            continue
        points = np.concatenate([np.asarray(p) for p in polygons])
        box = [
            int(points[:, 0].min()),
            int(points[:, 1].min()),
            int(points[:, 0].max() + 1),
            int(points[:, 1].max() + 1),
        ]
        bridge_used = any(
            region[int(a[1]), int(a[0])] and region[int(b[1]), int(b[0])]
            for a, b in inferred_bridges
        )
        output.append(
            {
                "kind": "scene",
                "class_name": "sidewalk",
                "confidence": max(x["confidence"] for x in members),
                "box": box,
                "xyxy": box,
                "segmentation_polygons": polygons,
                "geometry_source": "consolidated_model_masks",
                "merged_detection_count": len(members),
                "occlusion_bridge_inferred": bool(bridge_used),
                "review_required": True,
            }
        )
    return output
