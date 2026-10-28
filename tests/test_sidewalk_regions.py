"""Regression geometry for fragmented sidewalks, not field segmentation accuracy."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from src.sidewalk_regions import consolidate_sidewalks
from src.inference import annotate
from src.tracking import BoxTracker, EventAccumulator


class Data:
    def __init__(self, masks):
        self.masks = np.asarray(masks)

    def cpu(self):
        return self

    def numpy(self):
        return self.masks


def scene(masks, names=None):
    names = names or ["sidewalk"] * len(masks)
    items = []
    for i, (mask, name) in enumerate(zip(masks, names)):
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        items.append(
            {
                "kind": "scene",
                "class_name": name,
                "confidence": 0.8,
                "box": [x, y, x + w, y + h],
                "mask_index": i,
            }
        )
    return items, SimpleNamespace(masks=SimpleNamespace(data=Data(masks)))


class SidewalkTests(unittest.TestCase):
    def test_duplicate_and_overlapping_masks_become_one_region(self):
        a = np.zeros((120, 200), np.uint8)
        b = a.copy()
        a[10:90, 20:60] = 1
        b[50:110, 20:60] = 1
        items, result = scene([a, b])
        merged = consolidate_sidewalks(items, result, a.shape)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["merged_detection_count"], 2)
        self.assertEqual(merged[0]["box"], [20, 10, 60, 110])

    def test_fragments_behind_object_merge_without_mutating_model_masks(self):
        a = np.zeros((120, 200), np.uint8)
        b = a.copy()
        a[10:45, 20:60] = 1
        b[60:110, 20:60] = 1
        items, result = scene([a, b])
        before = result.masks.data.numpy().copy()
        merged = consolidate_sidewalks(items, result, a.shape, [[15, 40, 65, 65]])
        self.assertEqual(len(merged), 1)
        self.assertTrue(merged[0]["occlusion_bridge_inferred"])
        np.testing.assert_array_equal(before, result.masks.data.numpy())

    def test_large_unsupported_gap_stays_separate(self):
        a = np.zeros((120, 200), np.uint8)
        b = a.copy()
        a[10:45, 20:60] = 1
        b[60:110, 20:60] = 1
        items, result = scene([a, b])
        self.assertEqual(len(consolidate_sidewalks(items, result, a.shape)), 2)

    def test_visible_road_prevents_join_even_with_occluder(self):
        a = np.zeros((120, 200), np.uint8)
        b = a.copy()
        road = a.copy()
        a[10:45, 20:60] = 1
        b[60:110, 20:60] = 1
        road[45:60, :] = 1
        items, result = scene([a, b, road], ["sidewalk", "sidewalk", "road"])
        merged = consolidate_sidewalks(items, result, a.shape, [[15, 40, 65, 65]])
        self.assertEqual(sum(x["class_name"] == "sidewalk" for x in merged), 2)

    def test_parallel_sidewalks_are_not_combined_sideways(self):
        a = np.zeros((120, 200), np.uint8)
        b = a.copy()
        a[10:110, 20:40] = 1
        b[10:110, 55:75] = 1
        items, result = scene([a, b])
        self.assertEqual(
            len(consolidate_sidewalks(items, result, a.shape, [[35, 0, 60, 120]])), 2
        )

    def test_other_scene_classes_are_not_merged(self):
        a = np.zeros((120, 200), np.uint8)
        a[10:60, 10:60] = 1
        items, result = scene([a, a], ["sign", "sign"])
        self.assertEqual(consolidate_sidewalks(items, result, a.shape), items)

    def test_annotation_uses_outline_instead_of_rectangle(self):
        a = np.zeros((120, 200), np.uint8)
        a[10:110, 20:60] = 1
        items, result = scene([a])
        merged = consolidate_sidewalks(items, result, a.shape)
        with patch("src.inference.cv2.rectangle") as rectangle, patch(
            "src.inference.cv2.polylines", wraps=cv2.polylines
        ) as polylines:
            annotate(np.zeros((120, 200, 3), np.uint8), {"scene_segments": merged})
        rectangle.assert_not_called()
        polylines.assert_called_once()

    def test_video_tracks_one_region_when_mask_splits_behind_object(self):
        whole = np.zeros((120, 200), np.uint8)
        whole[10:110, 20:60] = 1
        top = whole.copy()
        top[45:] = 0
        bottom = whole.copy()
        bottom[:60] = 0
        tracker = BoxTracker()
        events = EventAccumulator()
        for time, masks in enumerate([[whole], [top, bottom], [whole]]):
            items, result = scene(masks)
            for track in tracker.update(
                consolidate_sidewalks(items, result, whole.shape, [[15, 40, 65, 65]])
            ):
                events.observe("scene", track, time)
        self.assertEqual(len(events.events), 1)
        self.assertEqual(next(iter(events.events.values()))["observations"], 3)


if __name__ == "__main__":
    unittest.main()
