"""Threshold selection and promotion guards. Test splits may never tune thresholds."""

import math


def tune_threshold(scores, labels, split="val", beta=1.0, missed_positives=0):
    """Scores/labels must be independently matched detection candidates, not model confidences as truth."""
    if split != "val":
        raise ValueError("Threshold selection is allowed on validation data only")
    if len(scores) != len(labels) or not scores or beta <= 0 or missed_positives < 0:
        raise ValueError("Invalid validation observations")
    if not all(math.isfinite(s) and 0 <= s <= 1 for s in scores) or not all(
        y in (0, 1) for y in labels
    ):
        raise ValueError("Invalid scores/labels")
    if sum(labels) == 0:
        raise ValueError("Positive validation observations required")
    results = []
    for threshold in sorted(set(scores)):
        tp = sum(s >= threshold and y == 1 for s, y in zip(scores, labels))
        fp = sum(s >= threshold and y == 0 for s, y in zip(scores, labels))
        fn = missed_positives + sum(
            s < threshold and y == 1 for s, y in zip(scores, labels)
        )
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn)
        f = (
            (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
            if precision + recall
            else 0
        )
        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f_beta": f,
            }
        )
    return max(results, key=lambda x: (x["f_beta"], x["recall"])), results


def promotion_decision(baseline, candidate, metric, data_fingerprint):
    required = ("split", "imgsz", "task")
    if any(baseline.get(k) != candidate.get(k) for k in required):
        raise ValueError("Evaluation protocols differ")
    if (
        not data_fingerprint
        or baseline.get("data_fingerprint") != data_fingerprint
        or candidate.get("data_fingerprint") != data_fingerprint
    ):
        raise ValueError("Same versioned evaluation data required")
    if (
        baseline.get("purpose") != "independent_test_verified"
        or candidate.get("purpose") != "independent_test_verified"
    ):
        return {"promote": False, "reason": "Independent test provenance is missing"}
    old = baseline["metrics"][metric]
    new = candidate["metrics"][metric]
    return {
        "promote": False,
        "metric_improved": new > old,
        "delta": new - old,
        "reason": "Manual activation requires per-class/privacy regression review, even when the aggregate improves",
    }
