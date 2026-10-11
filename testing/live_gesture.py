"""
Live check of the trained gesture MLP (models/gesture_mlp.pt).

Shows the per-frame top 3 predictions, plus a smoothed label: the most common
prediction over the last SMOOTH_FRAMES frames, shown only when its average
probability is above MIN_PROB.

Usage: python testing/live_gesture.py [--camera 1]
"""

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import relative_pose_features  # noqa: E402
from src.gesture_model import GestureClassifier  # noqa: E402
from src.hands import relative_pose  # noqa: E402

MODEL_PATH = "models/hand_landmarker.task"
SMOOTH_FRAMES = 10
MIN_PROB = 0.6

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--camera", type=int, default=1, help="cv2 camera index")
  args = p.parse_args()

  clf = GestureClassifier()
  history = deque(maxlen=SMOOTH_FRAMES)  # per-frame probability vectors

  cap = cv2.VideoCapture(args.camera)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)
  with HandLandmarker.create_from_options(options) as hands:
    start = time.monotonic()
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
      results = hands.detect_for_video(mp_image, int((time.monotonic() - start) * 1000))

      h, w = image.shape[:2]
      rp = relative_pose(results, (w, h))
      lines = []
      if rp is None:
        history.clear()
        lines.append("need two hands")
      else:
        probs = clf.predict_proba(relative_pose_features(rp))
        history.append(probs)
        votes = np.bincount([int(p.argmax()) for p in history], minlength=len(clf.labels))
        best = int(votes.argmax())
        avg = float(np.mean([p[best] for p in history]))
        smoothed = clf.labels[best] if avg >= MIN_PROB else "?"
        lines.append(f"{smoothed}  ({avg:.2f})")
        for i in probs.argsort()[::-1][:3]:
          lines.append(f"  {clf.labels[i]:<18} {probs[i]:.2f}")

      for hand_landmarks in results.hand_landmarks:
        drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style())

      # Flip the image horizontally for a selfie-view display.
      display = cv2.flip(image, 1)
      for k, text in enumerate(lines):
        scale = 1.2 if k == 0 else 0.7
        cv2.putText(display, text, (10, 40 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), 2)
      cv2.imshow('Live Gesture', display)
      if cv2.waitKey(5) & 0xFF == 27:
        break
  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  main()
