"""Ultralytics YOLO detector wrapper and local label-backed fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.utils.bbox import BBox, Detection

LOGGER = logging.getLogger(__name__)

CANONICAL_NAMES = [
    "person",
    "helmet",
    "safety_vest",
    "gloves",
    "safety_boots",
    "safety_goggles",
    "face_mask",
    "lab_coat",
]


class ModelUnavailableError(RuntimeError):
    """Raised when YOLO weights are required but unavailable."""


class PPEDetector:
    """Thin Ultralytics YOLO wrapper returning project Detection objects."""

    def __init__(
        self,
        weights_path: str | Path = "models/weights/ppe_best.pt",
        conf_threshold: float = 0.4,
        device: str | None = None,
    ) -> None:
        self.weights_path = Path(weights_path)
        self.conf_threshold = conf_threshold
        self.device = device or self._default_device()
        self.model: Any | None = None

        if not self.weights_path.exists():
            raise ModelUnavailableError(
                f"Model weights not found at {self.weights_path}. Run train.py first."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError(
                "ultralytics is not installed. Install requirements.txt first."
            ) from exc

        self.model = YOLO(str(self.weights_path))
        LOGGER.info("loaded_detector", extra={"weights_path": str(self.weights_path)})

    @staticmethod
    def _default_device() -> str:
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def predict(self, image: Image.Image | np.ndarray) -> list[Detection]:
        """Run YOLO inference and return canonical Detection objects."""
        if self.model is None:
            raise ModelUnavailableError("Detector model is not loaded.")

        results = self.model.predict(
            source=image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        names = results[0].names
        boxes = results[0].boxes
        for box in boxes:
            class_id = int(box.cls.item())
            class_name = str(names.get(class_id, CANONICAL_NAMES[class_id]))
            confidence = float(box.conf.item())
            x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        LOGGER.info(
            "predicted_detections",
            extra={"count": len(detections), "device": self.device},
        )
        return detections


class YoloLabelDetector:
    """Read YOLO labels as detections for local dataset QA and demos."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = names or CANONICAL_NAMES

    def predict_from_label_file(
        self,
        image: Image.Image,
        label_path: str | Path,
    ) -> list[Detection]:
        """Convert a YOLO label file into Detection objects for an image."""
        path = Path(label_path)
        if not path.exists():
            return []

        width, height = image.size
        detections: list[Detection] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, box_width, box_height = (float(value) for value in parts[1:5])
            class_name = self.names[class_id] if class_id < len(self.names) else str(class_id)
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    bbox=BBox.from_yolo(
                        x_center=x_center,
                        y_center=y_center,
                        width=box_width,
                        height=box_height,
                        image_width=width,
                        image_height=height,
                    ),
                )
            )
        return detections
