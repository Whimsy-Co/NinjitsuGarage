from src.camera import HostCam
from src.inference import YoloPersonDet
import cv2, time
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

    cam = HostCam(0)
    cam.open()

    while cam.is_open:
        frame = cam.read()
        if frame is None:
            break
        det.image = frame

        detected = det.detect_person()
        if detected is None:
            continue
        cx,cy = detected.center
        print(f"Person Detected at {cx:2f}, {cy:2f}")
        timeout = 0
        quit_requested = False

        with HandLandmarker.create_from_options(options) as hands:
            frame_ms = 0
            while True:
                # Grab a fresh frame each iteration (the old one was reused forever)
                frame = cam.read()
                if frame is None:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                frame_ms += 30
                results = hands.detect_for_video(mp_image, frame_ms)

                if not results.hand_landmarks:
                    timeout += 1
                    if timeout >= 120:
                        break
                else:
                    timeout = 0
                    for hand_landmarks in results.hand_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            HandLandmarksConnections.HAND_CONNECTIONS,
                            drawing_styles.get_default_hand_landmarks_style(),
                            drawing_styles.get_default_hand_connections_style())
                # Flip the image horizontally for a selfie-view display.
                cv2.imshow('MediaPipe Hands', cv2.flip(frame, 1))
                if cv2.waitKey(5) & 0xFF == 27:
                    quit_requested = True
                    break

        cv2.destroyAllWindows()
        time.sleep(PROCESS_DELAY)
        if quit_requested:
            break

    cam.close()












    pass

if __name__ == "__main__":
    
    main()