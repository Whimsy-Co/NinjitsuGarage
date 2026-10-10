"""
Credit to Mediapipe for this example, modified to the current tasks api
https://chuoling.github.io/mediapipe/solutions/hands#python-solution-api
"""

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

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)

# For webcam input:
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080.0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920.0)
with HandLandmarker.create_from_options(options) as hands:
  frame_ms = 0
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
    frame_ms += 60
    results = hands.detect_for_video(mp_image, frame_ms)

    # Draw the hand annotations on the image.
    if results.hand_landmarks:
      for hand_landmarks in results.hand_landmarks:
        drawing_utils.draw_landmarks(
            image,
            hand_landmarks,
            HandLandmarksConnections.HAND_CONNECTIONS,
            drawing_styles.get_default_hand_landmarks_style(),
            drawing_styles.get_default_hand_connections_style())
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
cv2.destroyAllWindows()
