"""Reusable, confidence-gated wound inference for the clinical API.

The segmentation model is an optional ROI proposal model.  When it cannot
produce a reliable crop, the classifier is run on the original image instead
of returning an empty result.  This module contains no dataset or test-set
access; it only accepts an image array supplied by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class CascadeConfig:
    segmenter_confidence: float = 0.25
    classifier_confidence: float = 0.60
    crop_margin: float = 0.15
    segmenter_imgsz: int = 224
    classifier_imgsz: int = 224
    device: str = "cpu"
    classification_source: str = "full_image"


def _class_name(model: Any, index: int) -> str:
    names = getattr(model, "names", {})
    return str(names[index] if isinstance(names, dict) else names[index])


def classify_image(model: Any, image: np.ndarray, config: CascadeConfig) -> tuple[Optional[str], Optional[float]]:
    """Return the top-1 class and confidence, without applying a threshold."""

    result = model.predict(
        source=image,
        imgsz=config.classifier_imgsz,
        device=config.device,
        verbose=False,
    )[0]
    probabilities = getattr(result, "probs", None)
    if probabilities is None:
        return None, None
    index = int(probabilities.top1)
    return _class_name(model, index), float(probabilities.top1conf)


def _segmentation_box(result: Any, width: int, height: int) -> tuple[Optional[list[int]], float, int]:
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xy", None) if masks is not None else None
    if polygons is None:
        return None, 0.0, 0
    valid = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            valid.append(points)
    if not valid:
        return None, 0.0, 0
    all_points = np.concatenate(valid, axis=0)
    x1 = max(0, int(np.floor(all_points[:, 0].min())))
    y1 = max(0, int(np.floor(all_points[:, 1].min())))
    x2 = min(width, int(np.ceil(all_points[:, 0].max() + 1)))
    y2 = min(height, int(np.ceil(all_points[:, 1].max() + 1)))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, 0
    boxes = getattr(result, "boxes", None)
    confidences = getattr(boxes, "conf", None) if boxes is not None else None
    confidence = float(confidences.max().detach().cpu().item()) if confidences is not None and len(confidences) else 0.0
    return [x1, y1, x2, y2], confidence, len(valid)


def _padded_box(box: list[int], width: int, height: int, margin: float) -> list[int]:
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    dx = int(round(box_width * margin))
    dy = int(round(box_height * margin))
    return [max(0, x1 - dx), max(0, y1 - dy), min(width, x2 + dx), min(height, y2 + dy)]


def infer_segmentation_cascade(
    image: np.ndarray,
    segmenter: Any,
    classifier: Any,
    config: CascadeConfig = CascadeConfig(),
) -> dict[str, Any]:
    """Run segmentation→crop→classification with full-image fallback."""

    height, width = image.shape[:2]
    segment_result = segmenter.predict(
        source=image,
        conf=config.segmenter_confidence,
        imgsz=config.segmenter_imgsz,
        device=config.device,
        verbose=False,
    )[0]
    raw_box, segment_confidence, instance_count = _segmentation_box(segment_result, width, height)
    crop_box = _padded_box(raw_box, width, height, config.crop_margin) if raw_box else None
    if crop_box:
        x1, y1, x2, y2 = crop_box
        crop = image[y1:y2, x1:x2]
    else:
        crop = None

    label: Optional[str] = None
    class_confidence: Optional[float] = None
    if config.classification_source == "full_image":
        # Domain-shift-safe production default: classify the original image
        # and use segmentation only to provide a visual ROI when available.
        label, class_confidence = classify_image(classifier, image, config)
        low_confidence = label is None or class_confidence is None or class_confidence < config.classifier_confidence
        if label is not None and not low_confidence:
            detections = 1 if crop_box else 0
            return {
                "mode": "full_image_primary",
                "fallback_used": False,
                "detected_boxes": detections,
                "segment_instances": instance_count,
                "segment_confidence": round(segment_confidence, 4) if raw_box else None,
                "crop_box": {"x1": crop_box[0], "y1": crop_box[1], "x2": crop_box[2], "y2": crop_box[3]} if crop_box else None,
                "label": label,
                "class_confidence": round(class_confidence, 4),
                "low_confidence": False,
                "classification_source": "full_image",
            }
        return {
            "mode": "full_image_primary",
            "fallback_used": False,
            "detected_boxes": 1 if crop_box else 0,
            "segment_instances": instance_count,
            "segment_confidence": round(segment_confidence, 4) if raw_box else None,
            "crop_box": {"x1": crop_box[0], "y1": crop_box[1], "x2": crop_box[2], "y2": crop_box[3]} if crop_box else None,
            "label": label,
            "class_confidence": round(class_confidence, 4) if class_confidence is not None else None,
            "low_confidence": True,
            "classification_source": "full_image",
        }

    if config.classification_source != "segmentation_crop":
        raise ValueError("classification_source must be full_image or segmentation_crop")

    if crop is not None and crop.size and segment_confidence >= config.segmenter_confidence:
        label, class_confidence = classify_image(classifier, crop, config)
        if label is not None and class_confidence is not None and class_confidence >= config.classifier_confidence:
            return {
                "mode": "segmentation_crop",
                "fallback_used": False,
                "detected_boxes": 1,
                "segment_instances": instance_count,
                "segment_confidence": round(segment_confidence, 4),
                "crop_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "label": label,
                "class_confidence": round(class_confidence, 4),
                "low_confidence": False,
                "classification_source": "segmentation_crop",
            }

    fallback_label, fallback_confidence = classify_image(classifier, image, config)
    low_confidence = fallback_label is None or fallback_confidence is None or fallback_confidence < config.classifier_confidence
    return {
        "mode": "full_image_fallback",
        "fallback_used": True,
        "detected_boxes": 0,
        "segment_instances": instance_count,
        "segment_confidence": round(segment_confidence, 4) if raw_box else None,
        "crop_box": None,
        "label": fallback_label,
        "class_confidence": round(fallback_confidence, 4) if fallback_confidence is not None else None,
        "low_confidence": low_confidence,
        "classification_source": "full_image",
    }
