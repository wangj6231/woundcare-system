from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from woundcare_inference import CascadeConfig, infer_segmentation_cascade


class FakeBoxes:
    def __init__(self, confidence: float):
        self.conf = torch.tensor([confidence], dtype=torch.float32)


class FakeMasks:
    def __init__(self, polygons):
        self.xy = polygons


class FakeSegmentationResult:
    def __init__(self, polygons, confidence: float = 0.9):
        self.masks = FakeMasks(polygons)
        self.boxes = FakeBoxes(confidence) if polygons else FakeBoxes(0.0)


class FakeClassificationResult:
    class Probs:
        top1 = 0
        top1conf = 0.95

    def __init__(self, confidence: float):
        self.probs = self.Probs()
        self.probs.top1conf = confidence


class FakeSegmenter:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [self.result]


class FakeClassifier:
    names = {0: "Abrasions"}

    def __init__(self, confidences):
        self.confidences = list(confidences)
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [FakeClassificationResult(self.confidences.pop(0))]


def test_reliable_segmentation_uses_crop():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    segmenter = FakeSegmenter(FakeSegmentationResult([np.array([[20, 20], [60, 20], [60, 60], [20, 60]])]))
    classifier = FakeClassifier([0.95])
    result = infer_segmentation_cascade(image, segmenter, classifier, CascadeConfig(classification_source="segmentation_crop"))
    assert result["mode"] == "segmentation_crop"
    assert result["fallback_used"] is False
    assert result["label"] == "Abrasions"
    assert len(classifier.calls) == 1


def test_missing_segmentation_uses_full_image_primary():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    segmenter = FakeSegmenter(FakeSegmentationResult([]))
    classifier = FakeClassifier([0.91])
    result = infer_segmentation_cascade(image, segmenter, classifier, CascadeConfig(classification_source="full_image"))
    assert result["mode"] == "full_image_primary"
    assert result["fallback_used"] is False
    assert result["crop_box"] is None
    assert len(classifier.calls) == 1


def test_full_image_primary_keeps_roi_for_display_only():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    segmenter = FakeSegmenter(FakeSegmentationResult([np.array([[20, 20], [60, 20], [60, 60], [20, 60]])]))
    classifier = FakeClassifier([0.95])
    result = infer_segmentation_cascade(image, segmenter, classifier, CascadeConfig(classification_source="full_image"))
    assert result["mode"] == "full_image_primary"
    assert result["fallback_used"] is False
    assert result["crop_box"] is not None
    assert len(classifier.calls) == 1


def test_low_crop_confidence_reclassifies_original_image():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    segmenter = FakeSegmenter(FakeSegmentationResult([np.array([[20, 20], [60, 20], [60, 60], [20, 60]])]))
    classifier = FakeClassifier([0.40, 0.92])
    result = infer_segmentation_cascade(image, segmenter, classifier, CascadeConfig(classifier_confidence=0.60, classification_source="segmentation_crop"))
    assert result["mode"] == "full_image_fallback"
    assert result["fallback_used"] is True
    assert result["class_confidence"] == 0.92
    assert len(classifier.calls) == 2


if __name__ == "__main__":
    test_reliable_segmentation_uses_crop()
    test_missing_segmentation_uses_full_image_primary()
    test_full_image_primary_keeps_roi_for_display_only()
    test_low_crop_confidence_reclassifies_original_image()
    print("PASS: woundcare inference unit tests")
