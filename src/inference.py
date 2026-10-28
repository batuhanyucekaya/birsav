"""Shared local model cache; serialize every call, including mutable predictors."""

import base64
import threading
import cv2
import numpy as np
from ultralytics import YOLO

from src.model_registry import model_path, manifest, version
from src.privacy import PrivacyProcessor
from src.scene_context import litter_context
from src.sidewalk_regions import consolidate_sidewalks

MODEL_LOCK = threading.RLock()
_models = {}
_model_keys = {}
LITTER_CLASSES = {
    "Plastic bag & wrapper",
    "Cigarette",
    "Bottle",
    "Bottle cap",
    "Can",
    "Other plastic",
    "Carton",
    "Cup",
    "Other",
}


def get_model(name, required=True):
    path = model_path(name, required)
    if path is None:
        return None
    config = manifest()[name]
    key = (str(path), config.get("sha256"))
    if name not in _models or _model_keys.get(name) != key:
        model = YOLO(str(path))
        if config.get("task") and model.task != config["task"]:
            raise ValueError(f"Model task mismatch: {name}")
        expected = {int(k): v for k, v in config.get("class_names", {}).items()}
        if expected and model.names != expected:
            raise ValueError(f"Model class mapping mismatch: {name}")
        _models[name] = model
        _model_keys[name] = key
    return _models[name]


def boxes(result, kind):
    if result is None or result.boxes is None:
        return []
    return [
        {
            "kind": kind,
            "class_name": result.names[int(c)],
            "confidence": float(p),
            "box": list(b),
            "xyxy": list(b),
        }
        for c, p, b in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.xyxy.tolist(),
        )
    ]


def container_predictions(frame):
    cfg = manifest()["container"]
    if not cfg["enabled"] or not (
        cfg.get("localization_validated") or cfg.get("allow_experimental")
    ):
        return (
            [],
            "Gerçek kutularla doğrulanmış konteyner modeli yok; bootstrap tahmini devre dışı.",
        )
    candidates = boxes(
        get_model("container").predict(frame, conf=0.25, imgsz=640, verbose=False)[0],
        "container",
    )
    h, w = frame.shape[:2]
    detections = []
    rejected = 0
    for item in candidates:
        if len(item["box"]) != 4 or not np.isfinite(item["box"]).all():
            rejected += 1
            continue
        x1, y1, x2, y2 = item["box"]
        x1, y1, x2, y2 = (
            max(0, int(x1)),
            max(0, int(y1)),
            min(w, int(x2)),
            min(h, int(y2)),
        )
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            rejected += 1
            continue
        # The bootstrap detector was trained on full-image boxes. A large frame
        # proposal cannot establish a container's location. This conservative
        # guard applies only to that unvalidated detector, not validated models.
        if not cfg.get("localization_validated") and bw * bh / (w * h) > 0.5:
            rejected += 1
            continue
        item.update(box=[x1, y1, x2, y2], xyxy=[x1, y1, x2, y2])
        crop = frame[y1:y2, x1:x2]
        item.update(
            class_name="unknown",
            state_confidence=None,
            review_required=True,
            experimental=not cfg.get("localization_validated", False),
            localization_validated=bool(cfg.get("localization_validated")),
        )
        if crop.size and min(crop.shape[:2]) >= 32:
            state = get_model("container_state").predict(crop, verbose=False)[0]
            p = float(state.probs.top1conf)
            item["state_confidence"] = p
            # Provisional abstention, not a validated occlusion/unknown classifier.
            if p >= 0.75:
                item["class_name"] = state.names[int(state.probs.top1)]
        detections.append(item)
    return (
        detections,
        "Konteyner tespiti ve boş/dolu/taşmış durum analizi aktif. "
        + (
            "Deneysel kutular ve durumlar doğrulanmadı; sonuçlar inceleme adayıdır."
            if not cfg.get("localization_validated")
            else "Durum eşikleri kalibre edilmedi; görünmeyen içerik manuel incelenmeli."
        )
        + (f" {rejected} geçersiz veya aşırı geniş kutu elendi." if rejected else ""),
    )


def analyze_frame(frame, include_lane=True, road_conf=0.10, litter_conf=0.25):
    """Caller must hold MODEL_LOCK. All semantic outputs remain review candidates."""
    road_result = get_model("road").predict(frame, conf=road_conf, verbose=False)[0]
    litter_result = get_model("litter").predict(frame, conf=litter_conf, verbose=False)[
        0
    ]
    scene_model = get_model("scene", False)
    scene_result = (
        scene_model.predict(
            frame, conf=0.08, imgsz=640, retina_masks=True, verbose=False
        )[0]
        if scene_model
        else None
    )
    scene = [
        {**x, "mask_index": index}
        for index, x in enumerate(boxes(scene_result, "scene"))
        if x["confidence"]
        >= {
            "road": 0.2,
            "sidewalk": 0.08,
            "lane_marking": 0.15,
            "trash_can": 0.3,
            "billboard": 0.12,
            "sign": 0.15,
        }.get(x["class_name"], 0.2)
    ]
    road = boxes(road_result, "road_damage")
    # Person/vehicle presence supplies an exclusion hint, not a legal decision.
    object_model = get_model("objects", False)
    objects = (
        object_model.predict(
            frame,
            conf=0.3,
            classes=[0, 1, 2, 3, 5, 7, 13, 56, 57, 58, 60],
            verbose=False,
        )[0]
        if object_model
        else None
    )
    exclusions = [x["box"] for x in boxes(objects, "context")]
    sidewalk_occluders = exclusions + [
        x["box"] for x in scene if x["class_name"] == "trash_can"
    ]
    scene = consolidate_sidewalks(scene, scene_result, frame.shape, sidewalk_occluders)
    litter = [
        x
        for x in boxes(litter_result, "litter_candidate")
        if x["class_name"] in LITTER_CLASSES
    ]
    for item in litter:
        item["source_object"] = item["class_name"]
        item["context"] = litter_context(
            item["box"], scene_result, frame.shape, exclusions
        )
    lane_model = get_model("lane", False) if include_lane else None
    lane = (
        boxes(
            lane_model.predict(frame, conf=0.2, imgsz=320, verbose=False)[0],
            "lane_marking",
        )
        if lane_model
        else []
    )
    containers, note = container_predictions(frame)
    return {
        "road_damage": road,
        "ground_litter": litter,
        "scene_segments": scene,
        "lane_markings": lane,
        "containers": containers,
        "container": containers[0] if len(containers) == 1 else None,
        "container_note": note,
        "issues": [
            {**x, "type": x["kind"], "review_required": True}
            for x in road + litter + containers
        ],
        "model_status": {k: version(k) for k in manifest()},
    }


def annotate(frame, result):
    output = frame.copy()
    for key, color in (
        ("road_damage", (0, 0, 255)),
        ("ground_litter", (0, 0, 255)),
        ("scene_segments", (180, 0, 180)),
        ("lane_markings", (0, 180, 0)),
        ("containers", (0, 180, 255)),
    ):
        for item in result.get(key, []):
            x1, y1, x2, y2 = map(int, item["box"])
            if key == "scene_segments" and item.get("segmentation_polygons"):
                contours = [
                    np.asarray(p, dtype=np.int32).reshape(-1, 1, 2)
                    for p in item["segmentation_polygons"]
                ]
                cv2.polylines(output, contours, True, color, 2)
            else:
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{item['class_name']} score={item['confidence']:.2f}"
            if key == "containers":
                state = {
                    "empty": "empty",
                    "full": "waste inside",
                    "overflowing": "overflowing",
                    "unknown": "state uncertain",
                }.get(item["class_name"], "state uncertain")
                state_score = item.get("state_confidence")
                label = f"container candidate: {state} det={item['confidence']:.2f}"
                if state_score is not None:
                    label += f" state={state_score:.2f}"
            cv2.putText(
                output,
                label,
                (max(0, x1), max(15, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
            )
    return output


def jpeg_payload(frame):
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Sonuç görseli üretilemedi")
    return base64.b64encode(encoded.tobytes()).decode("ascii")
