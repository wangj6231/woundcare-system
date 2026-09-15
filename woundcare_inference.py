"""Reusable, confidence-gated wound inference for the clinical API.

The segmentation model is an optional ROI proposal model.  When it cannot
produce a reliable crop, the classifier is run on the original image instead
of returning an empty result.  This module contains no dataset or test-set
access; it only accepts an image array supplied by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
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
    # Explicitly retain the existing Ultralytics NMS default. Calibration is a
    # separate development experiment, never inferred from sealed test scores.
    segmenter_iou: float = 0.70

    def __post_init__(self) -> None:
        for value in (self.segmenter_confidence, self.classifier_confidence, self.segmenter_iou):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("confidence and NMS thresholds must be finite in [0, 1]")
        if not math.isfinite(self.crop_margin) or not 0 <= self.crop_margin <= 1:
            raise ValueError("crop_margin must be finite in [0, 1]")
        for size in (self.segmenter_imgsz, self.classifier_imgsz):
            if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
                raise ValueError("inference image sizes must be positive integers")
        if self.classification_source not in {"full_image", "segmentation_crop"}:
            raise ValueError("classification_source must be full_image or segmentation_crop")


def _validate_bgr_image(image: np.ndarray) -> None:
    """Require OpenCV-style HWC uint8 BGR; channel semantics cannot be guessed.

    A caller holding RGB must convert explicitly before calling this module.
    Do not silently convert here: the API already uses cv2.imdecode (BGR).
    """
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3
            or image.shape[2] != 3 or min(image.shape[:2]) == 0):
        raise ValueError("expected a nonempty HWC uint8 BGR image")


def _class_name(model: Any, index: int) -> str:
    names = getattr(model, "names", {})
    return str(names[index] if isinstance(names, dict) else names[index])


def classify_image(model: Any, image: np.ndarray, config: CascadeConfig) -> tuple[Optional[str], Optional[float]]:
    """Return a finite top-1 class/confidence for a uint8 BGR image."""

    _validate_bgr_image(image)
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
    confidence = float(probabilities.top1conf)
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return None, None
    return _class_name(model, index), confidence


def _segmentation_box(result: Any, width: int, height: int) -> tuple[Optional[list[int]], float, int]:
    masks = getattr(result, "masks", None)
    polygons = getattr(masks, "xy", None) if masks is not None else None
    if polygons is None:
        return None, 0.0, 0
    boxes = getattr(result, "boxes", None)
    confidences = getattr(boxes, "conf", None) if boxes is not None else None
    if confidences is None or len(confidences) != len(polygons):
        return None, 0.0, 0
    valid, valid_confidences = [], []
    for index, polygon in enumerate(polygons):
        points = np.asarray(polygon, dtype=np.float32)
        confidence = float(confidences[index].detach().cpu().item())
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or not math.isfinite(confidence)
                or not 0 <= confidence <= 1):
            continue
        points = points.copy()
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        area2 = np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                       - np.roll(points[:, 0], -1) * points[:, 1])
        if abs(float(area2)) > 1e-6:
            valid.append(points)
            valid_confidences.append(confidence)
    if not valid:
        return None, 0.0, 0
    all_points = np.concatenate(valid, axis=0)
    x1 = max(0, int(np.floor(all_points[:, 0].min())))
    y1 = max(0, int(np.floor(all_points[:, 1].min())))
    x2 = min(width, int(np.ceil(all_points[:, 0].max() + 1)))
    y2 = min(height, int(np.ceil(all_points[:, 1].max() + 1)))
    if x2 <= x1 or y2 <= y1:
        return None, 0.0, 0
    return [x1, y1, x2, y2], max(valid_confidences), len(valid)


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
    """Run ROI proposal and classification on a uint8 BGR image.

    Full-image classification stays the default; a displayed ROI is not proof
    of a correct wound localization or of clinically validated performance.
    """

    _validate_bgr_image(image)
    height, width = image.shape[:2]
    segment_result = segmenter.predict(
        source=image,
        conf=config.segmenter_confidence,
        iou=config.segmenter_iou,
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
        # Development-supported default: classify the original image
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
