from src.camera import HostCam
from src.inference import YoloPersonDet
from pathlib import Path
from collections import deque
import cv2, time, sys
import numpy as np
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import relative_pose_features  # noqa: E402
from src.gesture_model import GestureClassifier  # noqa: E402
from src.hands import relative_pose  # noqa: E402
from src.sequence import SequenceMatcher  # noqa: E402

sequence = ["poc", "dog", "gloving"]

HOLD_TIME = 0.5     # seconds each move must be held to count
STEP_TIMEOUT = 10.0  # seconds allowed between moves before progress resets
NO_HANDS_TIMEOUT = 10.0  # seconds without hands before going back to person detection
MIN_PROB = 0.8

PROCESS_DELAY = 10 # how long to delay between detection attempts
MODEL_PATH = "models/hand_landmarker.task"

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=4,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5)


def main():

    det = YoloPersonDet()
    det.load_model()

    cam = HostCam(1)
    cam.open()

    clf = GestureClassifier()
    matcher = SequenceMatcher(sequence, clf.labels, hold_s=HOLD_TIME,
                              step_timeout_s=STEP_TIMEOUT, min_prob=MIN_PROB)

    start = time.monotonic()
    camera_failures = 0

    while cam.is_open:
        
        frame = cam.read()

        if frame is None:
            camera_failures += 1
            if camera_failures> 10: break
            continue
        camera_failures = 0

        det.image = frame

        detected = det.detect_person()
        if detected is None:
            continue
        cx,cy = detected.center
        print(f"Person Detected at {cx:2f}, {cy:2f}")

        # messes with distance math, likely not worth keeping
        
        # h, w = frame.shape[:2]
        # x1, y1, x2, y2 = detected.xyxy

        # half_w = (x2 - x1) / 2 * 1.35
        # half_h = (y2 - y1) / 2 * 1.2

        # x1, x2 = int(max(0, cx-half_w)), int(max(w, cx + half_w))
        # y1, y2 = int(max(0, cy - half_h)), int(max(h, cy + half_h))


        quit_requested = False
        matcher.reset()

        with HandLandmarker.create_from_options(options) as hands:
            # detection loop: one frame per iteration, the matcher tracks the sequence
            last_hands = time.monotonic()
            while True:
                image = cam.read()
                if image is None:
                    camera_failures += 1
                    if camera_failures > 10:
                        break
                    continue
                camera_failures = 0

                # cropped = frame[y1:y2, x1:x2]

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                now = time.monotonic()
                results = hands.detect_for_video(mp_image, int((now - start) * 1000))

                if results.hand_landmarks:
                    last_hands = now
                elif now - last_hands > NO_HANDS_TIMEOUT:
                    break

                h, w = image.shape[:2]
                rp = relative_pose(results, (w, h))
                probs = None if rp is None else clf.predict_proba(relative_pose_features(rp))

                event = matcher.update(probs, now)
                if event == "step":
                    print(f"{sequence[matcher.step - 1]} ({matcher.step}/{len(sequence)})")
                elif event == "reset":
                    print("Sequence reset")
                elif event == "complete":
                    print(f"Sequence complete: {' -> '.join(sequence)}")
                    break  # trigger the action here

                for hand_landmarks in results.hand_landmarks:
                    drawing_utils.draw_landmarks(
                        image,
                        hand_landmarks,
                        HandLandmarksConnections.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style())

                # Flip the image horizontally for a selfie-view display.
                display = cv2.flip(image, 1)
                seen = matcher.smoothed() or "-"
                cv2.putText(display, f"next: {matcher.expected} ({matcher.step}/{len(sequence)})  seen: {seen}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.imshow('MediaPipe Hands', display)
                if cv2.waitKey(5) & 0xFF == 27:
                    quit_requested = True
                    break

        cv2.destroyAllWindows()
        time.sleep(PROCESS_DELAY)
        if quit_requested:
            break

    cam.close()

if __name__ == "__main__":
    
    main()