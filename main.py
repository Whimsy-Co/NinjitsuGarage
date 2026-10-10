from camera import HostCam
from inference import YoloPersonDet

def main():

    det = YoloPersonDet()
    det.load_model()

    cam = HostCam(0)
    cam.open()

    while True:
        det.image = cam.read()
        detected = det.detect_person()
        if detected is None:
            continue
        cx,cy = detected.center
        print(f"Person Detected at {cx:2f}, {cy:2f}")







    pass

if __name__ == "__main__":
    
    main()