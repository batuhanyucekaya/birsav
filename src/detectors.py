"""Kent sorunu tespit modülleri için ortak veri sözleşmesi ve kayıt defteri."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Detection:
    issue_type: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None
    note: str = ""


Detector = Callable[[object], list[Detection]]


class DetectorRegistry:
    def __init__(self) -> None:
        self._detectors: dict[str, Detector] = {}

    def register(self, issue_type: str, detector: Detector) -> None:
        self._detectors[issue_type] = detector

    def run(self, frame: object) -> list[Detection]:
        detections: list[Detection] = []
        for detector in self._detectors.values():
            detections.extend(detector(frame))
        return detections

    def names(self) -> list[str]:
        return sorted(self._detectors)


def unavailable_detector(issue_type: str, reason: str) -> Detector:
    def detect(_: object) -> list[Detection]:
        return [Detection(issue_type, 0.0, note=f"Model bekleniyor: {reason}")]
    return detect


def build_demo_registry() -> DetectorRegistry:
    registry = DetectorRegistry()
    registry.register("asfalt_bozuklugu", unavailable_detector("asfalt_bozuklugu", "RDD2022 inference akışına bağlı"))
    registry.register("konteyner_tasmasi", unavailable_detector("konteyner_tasmasi", "dolu/boş etiketli veri gerekiyor"))
    registry.register("sokak_isgali", unavailable_detector("sokak_isgali", "kaldırım alanı ve masa/sandalye etiketleri gerekiyor"))
    registry.register("reklam_panosu", unavailable_detector("reklam_panosu", "pano/tabela etiketli veri gerekiyor"))
    return registry

