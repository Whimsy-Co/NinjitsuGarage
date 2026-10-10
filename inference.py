from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0  # COCO class index for "person"


@dataclass(frozen=True)
class Detection:
    """Bounding box in pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


class YoloPersonDet:
    """A single image (BGR ndarray, as produced by cv2) with inference helpers."""

    _model: Optional[YOLO] = None

    def __init__(self, image: np.ndarray = None):
        self.image = image if image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        self._person: Optional[Detection] = None
        self._inferred = False

    @classmethod
    def load_model(cls, weights: str = "yolov8n.pt") -> YOLO:
        """Load the shared YOLO model (once). Call early to avoid first-frame latency."""
        if cls._model is None:
            cls._model = YOLO(weights)
        return cls._model

    def detect_person(self, conf: float = 0.25) -> Optional[Detection]:
        """Run YOLO and return the highest-confidence person, or None."""
        model = self.load_model()
        results = model.predict(self.image, classes=[PERSON_CLASS_ID], conf=conf, verbose=False)
        boxes = results[0].boxes
        self._inferred = True
        if len(boxes) == 0:
            self._person = None
        else:
            best = int(boxes.conf.argmax())
            x1, y1, x2, y2 = boxes.xyxy[best].tolist()
            self._person = Detection(x1, y1, x2, y2, float(boxes.conf[best]))
        return self._person

    def has_person(self, conf: float = 0.25) -> bool:
        return self.detect_person(conf) is not None

    @property
    def person(self) -> Optional[Detection]:
        """Result of the last detect_person() call (runs it if not yet run)."""
        if not self._inferred:
            self.detect_person()
        return self._person
