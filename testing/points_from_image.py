"""
Credit to Mediapipe for this example, modified to the current tasks api
https://chuoling.github.io/mediapipe/solutions/hands#python-solution-api

Opens the webcam and, once two hands have been continuously detected for
HOLD_SECONDS, saves a snapshot (.jpg) and a text report (.txt) describing the
hands relative to each other and in camera space, then exits.
"""

import math
import time
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
    RunningMode,
    drawing_styles,
    drawing_utils,
)

MODEL_PATH = "models/hand_landmarker.task"
OUTPUT_DIR = Path("data")
HOLD_SECONDS = 5.0
REQUIRED_HANDS = 1
# Webcam diagonal field of view, used for the pinhole model (1280x720 sensor).
# Works out to roughly 71 deg horizontal / 44 deg vertical.
DFOV_DEG = 78.3

LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
]
# Bones used to estimate distance: (wrist -> middle MCP), (index MCP -> pinky MCP)
SCALE_PAIRS = [(0, 9), (5, 17)]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)


def estimate_camera_xyz(landmarks, world_landmarks, w, h, focal_px):
  """Pinhole estimate of the wrist position in camera space (meters).

  Camera frame: +X right, +Y down, +Z forward (away from the camera).
  Depth comes from comparing bone lengths in pixels vs. the metric bone
  lengths in world_landmarks, so it is approximate.
  """
  zs = []
  for i, j in SCALE_PAIRS:
    px = math.hypot((landmarks[i].x - landmarks[j].x) * w,
                    (landmarks[i].y - landmarks[j].y) * h)
    m = math.dist((world_landmarks[i].x, world_landmarks[i].y, world_landmarks[i].z),
                  (world_landmarks[j].x, world_landmarks[j].y, world_landmarks[j].z))
    if px > 1e-6:
      zs.append(focal_px * m / px)
  if not zs:
    return None
  z = sum(zs) / len(zs)
  u, v = landmarks[0].x * w, landmarks[0].y * h
  return ((u - w / 2) * z / focal_px, (v - h / 2) * z / focal_px, z)


def write_report(path, results, w, h):
  focal_px = (math.hypot(w, h) / 2) / math.tan(math.radians(DFOV_DEG) / 2)
  lines = [
      f"Capture: {datetime.now().isoformat(timespec='seconds')}",
      f"Image size: {w}x{h} px",
      f"Assumed DFOV: {DFOV_DEG} deg -> focal length {focal_px:.1f} px",
      "Camera frame: +X right, +Y down, +Z forward, meters (approximate)",
      "Handedness is as seen by MediaPipe on the un-mirrored image.",
      "",
  ]

  wrists = []
  for idx, (lm, wlm, hd) in enumerate(
      zip(results.hand_landmarks, results.hand_world_landmarks, results.handedness)):
    label = f"{hd[0].category_name} (score {hd[0].score:.2f})"
    xyz = estimate_camera_xyz(lm, wlm, w, h, focal_px)
    wrists.append((hd[0].category_name, lm[0], xyz))

    lines.append(f"=== Hand {idx}: {label} ===")
    if xyz:
      lines.append(f"Wrist in camera space (m): X={xyz[0]:+.3f} Y={xyz[1]:+.3f} Z={xyz[2]:.3f}")
    lines.append("Landmarks: name | image px (x, y) | normalized z (wrist-relative, ~x scale) "
                 "| hand-centered world (m)")
    for i, (p, q) in enumerate(zip(lm, wlm)):
      lines.append(
          f"  {i:2d} {LANDMARK_NAMES[i]:<11} | ({p.x * w:7.1f}, {p.y * h:7.1f}) "
          f"| {p.z:+.4f} | ({q.x:+.4f}, {q.y:+.4f}, {q.z:+.4f})")
    lines.append("")

  if len(wrists) >= 2:
    (name_a, a, xyz_a), (name_b, b, xyz_b) = wrists[0], wrists[1]
    lines.append(f"=== Relative: {name_a} -> {name_b} (wrist to wrist) ===")
    lines.append(f"Image offset px: dx={(b.x - a.x) * w:+.1f} dy={(b.y - a.y) * h:+.1f} "
                 f"dist={math.hypot((b.x - a.x) * w, (b.y - a.y) * h):.1f}")
    if xyz_a and xyz_b:
      d = tuple(q - p for p, q in zip(xyz_a, xyz_b))
      lines.append(f"Camera-space offset m: dX={d[0]:+.3f} dY={d[1]:+.3f} dZ={d[2]:+.3f} "
                   f"dist={math.sqrt(sum(c * c for c in d)):.3f}")
  path.write_text("\n".join(lines) + "\n")


OUTPUT_DIR.mkdir(exist_ok=True)

# For webcam input:
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280.0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720.0)
with HandLandmarker.create_from_options(options) as hands:
  start = time.monotonic()
  detected_since = None
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # The Tasks API expects an mp.Image in SRGB format.
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    # VIDEO mode needs a monotonically increasing timestamp in milliseconds.
    now = time.monotonic()
    results = hands.detect_for_video(mp_image, int((now - start) * 1000))

    # Track how long the required number of hands has been continuously seen.
    if len(results.hand_landmarks) >= REQUIRED_HANDS:
      if detected_since is None:
        detected_since = now
    else:
      detected_since = None
    held = 0.0 if detected_since is None else now - detected_since

    # Draw the hand annotations on the image.
    if results.hand_landmarks:
      for hand_landmarks in results.hand_landmarks:
        drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style())

    if held >= HOLD_SECONDS:
      h, w = image.shape[:2]
      stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      cv2.imwrite(str(OUTPUT_DIR / f"hands_{stamp}.jpg"), image)
      write_report(OUTPUT_DIR / f"hands_{stamp}.txt", results, w, h)
      print(f"Saved capture to {OUTPUT_DIR}/hands_{stamp}.(jpg|txt)")
      break

    # Flip the image horizontally for a selfie-view display.
    display = cv2.flip(image, 1)
    cv2.putText(display, f"hands: {len(results.hand_landmarks)}/{REQUIRED_HANDS}  hold: {held:.1f}/{HOLD_SECONDS:.0f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imshow('MediaPipe Hands', display)
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
