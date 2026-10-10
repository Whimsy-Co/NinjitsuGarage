from abc import ABC, abstractmethod
from typing import Optional, Tuple
import cv2


class Feed:
    @abstractmethod
    def open(self)-> None: ...

    @abstractmethod 
    def close(self)-> None: ...

    @abstractmethod
    def read(self)-> None: ...

    @property
    @abstractmethod
    def resolution(self)-> Tuple[int, int]:...

    @property
    @abstractmethod
    def fps(self)->float: ...

    def __enter__(self):
        self.open()
        return self

    def __exit__(self):
        self.close()
        return self

    def __iter__(self):
        while (frame:=self.read()) is not None:
            yield frame


class HostCam(Feed):
    """
    Webcam, file, etc that are directly accessed by openCV
    """
    def __init__(self, source: int|str, 
                 resolution: Optional[Tuple[int,int]] = None, 
                 fps: Optional[float]= None):
        self._source = source
        self._res = resolution
        self._fps = fps
        self._cap = Optional[cv2.VideoCapture] = None

    def open(self)->None:
        self._cap = cv2.VideoCapture(self._source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open cv2 stream")
        if self._res:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._res[0])
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._res[1])
        if self._fps:
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
    def read(self)->None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def close(self)->None:
        if self.cap is not None:
            self._cap.release()


if __name__ == "main":

    # Open the video file
    video = cv2.VideoCapture(0)

    # Get the FPS
    fps = video.get(cv2.CAP_PROP_FPS)

    print("Frames per second using video.get(cv2.CAP_PROP_FPS):", fps)

    # Release the video capture object
    video.release()
