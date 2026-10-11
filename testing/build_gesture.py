"""
Record one two-hand gesture from several distances and angles, for training.

The gesture is held for HOLD_SECONDS at each entry in VIEWS (head-on, closer,
farther, then the same turned left and turned right), with a BREAK_SECONDS
countdown before each to get into position. Every frame with two hands during a
hold is converted into the right hand's frame (src.hands.relative_pose) and kept
as a sample. The hold clock only runs while two hands are visible. When all
views are done, you're asked for the gesture's name in the terminal and the
samples are saved to data/gestures/<name>_<stamp>.npz.

Usage: python testing/build_gesture.py
"""

import argparse
import sys
import time
from datetime import datetime
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
from src.hands import relative_pose  # noqa: E402

MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_DIR = Path("data/gestures")
HOLD_SECONDS = 5.0
BREAK_SECONDS = 5.0
REQUIRED_HANDS = 2
# Where to stand for each hold, in order.
VIEWS = [
    "head-on",
    "head-on, closer",
    "head-on, farther",
    "turned left",
    "turned left, closer",
    "turned left, farther",
    "turned right",
    "turned right, closer",
    "turned right, farther",
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)


def parse_args():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument("--camera", type=int, default=0, help="cv2 camera index")
  return p.parse_args()


def ask_name():
  while True:
    name = input("Gesture name: ").strip().replace(" ", "_")
    if name:
      return name


def save(path, name, samples):
  stack = lambda key: np.stack([s[key] for s in samples])
  np.savez(
      path,
      name=name,
      views=np.array(VIEWS),
      hold_seconds=HOLD_SECONDS,
      view_index=np.array([s["view"] for s in samples]),  # index into views
      t=np.array([s["t"] for s in samples]),              # hold-clock seconds within that view
      anchor_points=stack("anchor_points"),               # (S, 21, 3) right hand in its own frame
      other_points=stack("other_points"),                 # (S, 21, 3) left hand in right-hand frame
      translation=stack("translation"),                   # (S, 3) meters
      normalized_translation=stack("normalized_translation"),
      rotation=stack("rotation"),                         # (S, 3, 3)
      scale=stack("scale"),                               # (S,) meters
  )


def main():
  args = parse_args()
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  cap = cv2.VideoCapture(args.camera)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)

  samples = []
  view = 0
  phase = "break"
  phase_start = time.monotonic()
  held = 0.0          # two-hand time accumulated in the current hold
  last = None         # previous frame time, for advancing the hold clock
  view_samples = 0
  aborted = False

  with HandLandmarker.create_from_options(options) as hands:
    start = time.monotonic()
    while cap.isOpened():
      success, image = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
      now = time.monotonic()
      results = hands.detect_for_video(mp_image, int((now - start) * 1000))
      n_hands = len(results.hand_landmarks)

      if phase == "break":
        remaining = BREAK_SECONDS - (now - phase_start)
        status = f"{view + 1}/{len(VIEWS)} next: {VIEWS[view]}  {max(remaining, 0):.1f}s"
        if remaining <= 0:
          phase, held, last, view_samples = "hold", 0.0, None, 0
      else:
        rp = None
        if n_hands >= REQUIRED_HANDS:
          h, w = image.shape[:2]
          rp = relative_pose(results, (w, h))
        if rp is not None:
          if last is not None:
            held += now - last
          last = now
          samples.append({
              "view": view, "t": held,
              "anchor_points": rp.anchor_points, "other_points": rp.other_points,
              "translation": rp.translation, "normalized_translation": rp.normalized_translation,
              "rotation": rp.rotation, "scale": rp.scale,
          })
          view_samples += 1
        else:
          last = None  # pause the hold clock until both hands are back
        status = (f"{view + 1}/{len(VIEWS)} {VIEWS[view]}: HOLD  "
                  f"{held:.1f}/{HOLD_SECONDS:.0f}s  samples: {view_samples}")
        if held >= HOLD_SECONDS:
          print(f"{VIEWS[view]}: {view_samples} samples")
          view += 1
          if view == len(VIEWS):
            break
          phase, phase_start = "break", now

      for hand_landmarks in results.hand_landmarks:
        drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style())

      # Flip the image horizontally for a selfie-view display.
      display = cv2.flip(image, 1)
      color = (0, 255, 0) if phase == "hold" and last is not None else (0, 200, 255)
      cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
      cv2.putText(display, f"hands: {n_hands}/{REQUIRED_HANDS}", (10, 65),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
      cv2.imshow('Build Gesture', display)
      if cv2.waitKey(5) & 0xFF == 27:
        aborted = True
        break

  cap.release()
  cv2.destroyAllWindows()

  if aborted or view < len(VIEWS):
    print("Aborted, nothing saved.")
    return
  name = ask_name()
  stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  path = OUTPUT_DIR / f"{name}_{stamp}.npz"
  save(path, name, samples)
  print(f"Saved {len(samples)} samples across {len(VIEWS)} views to {path}")


if __name__ == "__main__":
  main()
