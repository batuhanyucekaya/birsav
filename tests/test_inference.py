"""Geometry and gating checks with explicit fake models, not accuracy evidence."""

from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from src.scene_context import litter_context
from src.inference import container_predictions
from src.model_registry import model_path


class TensorLike:
    def __init__(self, value):
        self.value = np.array(value)

    def tolist(self):
        return self.value.tolist()

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class InferenceTests(unittest.TestCase):
    def test_unvalidated_full_scene_box_is_rejected_before_classification(self):
        result = SimpleNamespace(
            names={0: "empty"},
            boxes=SimpleNamespace(
                cls=TensorLike([0]),
                conf=TensorLike([0.25]),
                xyxy=TensorLike([[2, 5, 98, 95]]),
            ),
        )
        with patch(
            "src.inference.manifest",
            return_value={
                "container": {
                    "enabled": True,
                    "localization_validated": False,
                    "allow_experimental": True,
                }
            },
        ), patch("src.inference.get_model") as load:
            load.return_value.predict.return_value = [result]
            detections, note = container_predictions(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(detections, [])
        load.assert_called_once_with("container")
        self.assertIn("1 geçersiz", note)

    def test_validated_close_container_is_not_rejected_for_large_size(self):
        result = SimpleNamespace(
            names={0: "container"},
            boxes=SimpleNamespace(
                cls=TensorLike([0]),
                conf=TensorLike([0.9]),
                xyxy=TensorLike([[0, 0, 100, 100]]),
            ),
        )

        class Detector:
            def predict(self, *args, **kwargs):
                return [result]

        class Classifier:
            def predict(self, crop, **kwargs):
                return [
                    SimpleNamespace(
                        names={0: "empty"}, probs=SimpleNamespace(top1=0, top1conf=0.95)
                    )
                ]

        with patch(
            "src.inference.manifest",
            return_value={
                "container": {"enabled": True, "localization_validated": True}
            },
        ), patch(
            "src.inference.get_model",
            side_effect=lambda name: (
                Detector() if name == "container" else Classifier()
            ),
        ):
            detections, _ = container_predictions(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["class_name"], "empty")

    def test_unknown_overlay_distinguishes_state_and_detection_scores(self):
        from src.inference import annotate

        item = {
            "box": [10, 10, 50, 50],
            "class_name": "unknown",
            "confidence": 0.25,
            "state_confidence": 0.4,
        }
        with patch("src.inference.cv2.putText") as text:
            annotate(np.zeros((100, 100, 3), np.uint8), {"containers": [item]})
        label = text.call_args.args[1]
        self.assertIn("state uncertain", label)
        self.assertIn("det=0.25", label)
        self.assertIn("state=0.40", label)
        self.assertNotIn("unknown score", label)

    def test_explicit_experimental_activation_runs_unvalidated_detector(self):
        result = SimpleNamespace(
            names={},
            boxes=SimpleNamespace(
                cls=TensorLike([]), conf=TensorLike([]), xyxy=TensorLike([])
            ),
        )
        with patch(
            "src.inference.manifest",
            return_value={
                "container": {
                    "enabled": True,
                    "localization_validated": False,
                    "allow_experimental": True,
                }
            },
        ), patch("src.inference.get_model") as load:
            load.return_value.predict.return_value = [result]
            detections, note = container_predictions(np.zeros((100, 100, 3), np.uint8))
        load.assert_called_once_with("container")
        self.assertEqual(detections, [])
        self.assertIn("aktif", note)

    def test_model_task_contract_rejects_wrong_architecture_task(self):
        from src.inference import get_model

        with patch("src.inference.model_path", return_value=Path("fake.pt")), patch(
            "src.inference.manifest",
            return_value={"contract-test": {"task": "detect", "class_names": {}}},
        ), patch(
            "src.inference.YOLO", return_value=SimpleNamespace(task="segment", names={})
        ):
            with self.assertRaises(ValueError):
                get_model("contract-test")

    def test_mask_support_is_only_a_candidate(self):
        mask = np.zeros((100, 100), dtype=float)
        mask[60:90, :] = 1
        scene = SimpleNamespace(
            names={0: "sidewalk"},
            boxes=SimpleNamespace(cls=TensorLike([0])),
            masks=SimpleNamespace(data=TensorLike([mask])),
        )
        result = litter_context([30, 50, 50, 80], scene, (100, 100))
        self.assertEqual(result["status"], "surface_supported_candidate")
        self.assertTrue(result["review_required"])
        result = litter_context([30, 50, 50, 80], scene, (100, 100), [[25, 45, 55, 85]])
        self.assertEqual(result["status"], "unverified")

    def test_misaligned_masks_are_not_silently_resized(self):
        scene = SimpleNamespace(
            names={0: "road"},
            boxes=SimpleNamespace(cls=TensorLike([0])),
            masks=SimpleNamespace(data=TensorLike([np.ones((64, 64))])),
        )
        self.assertEqual(
            litter_context([30, 50, 50, 80], scene, (100, 100))["status"], "unverified"
        )

    def test_bootstrap_detector_never_returns_a_fake_container(self):
        with patch(
            "src.inference.manifest",
            return_value={
                "container": {"enabled": False, "localization_validated": False}
            },
        ):
            detections, note = container_predictions(np.zeros((100, 100, 3), np.uint8))
        self.assertEqual(detections, [])

    def test_each_container_is_classified_on_its_own_crop(self):
        frame = np.zeros((120, 120, 3), np.uint8)
        frame[20:60, 20:60] = 50
        frame[70:110, 70:110] = 200
        result = SimpleNamespace(
            names={0: "container"},
            boxes=SimpleNamespace(
                cls=TensorLike([0, 0]),
                conf=TensorLike([0.9, 0.8]),
                xyxy=TensorLike([[20, 20, 60, 60], [70, 70, 110, 110]]),
            ),
        )
        crops = []

        class Detector:
            def predict(self, *a, **kw):
                return [result]

        class Classifier:
            def predict(self, crop, **kw):
                crops.append(crop.copy())
                value = 0.95 if crop.mean() < 100 else 0.4
                return [
                    SimpleNamespace(
                        names={0: "empty"},
                        probs=SimpleNamespace(top1=0, top1conf=value),
                    )
                ]

        with patch(
            "src.inference.manifest",
            return_value={
                "container": {"enabled": True, "localization_validated": True}
            },
        ), patch(
            "src.inference.get_model",
            side_effect=lambda name: (
                Detector() if name == "container" else Classifier()
            ),
        ):
            predictions, _ = container_predictions(frame)
        self.assertEqual([p["class_name"] for p in predictions], ["empty", "unknown"])
        self.assertEqual([c.shape for c in crops], [(40, 40, 3), (40, 40, 3)])
        self.assertEqual([c.mean() for c in crops], [50, 200])

    def test_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "model.pt").write_bytes(b"changed")
            with patch("src.model_registry.ROOT", Path(folder)), patch(
                "src.model_registry.manifest",
                return_value={
                    "test": {
                        "enabled": True,
                        "path": "model.pt",
                        "sha256": "not-the-hash",
                        "status": "test",
                    }
                },
            ):
                with self.assertRaises(ValueError):
                    model_path("test")


if __name__ == "__main__":
    unittest.main()
