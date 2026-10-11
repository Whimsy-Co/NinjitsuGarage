from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

# Webcam diagonal field of view, used to build the pinhole camera matrix.
DFOV_DEG = 78.3

WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9
PINKY_MCP = 17
PALM = [WRIST, INDEX_MCP, MIDDLE_MCP, 13, PINKY_MCP]  # wrist + 4 knuckles


@dataclass(frozen=True)
class RelativePose:
    """Hand B expressed in hand A's palm frame (right-handed, meters).

    Frame A: origin at A's palm center, +X toward the pinky (index MCP -> pinky MCP),
    +Y toward the fingers (wrist -> middle MCP), +Z = X x Y. For a right hand, +Z
    points out of the palm.
    """
    anchor_label: str
    other_label: str
    anchor_points: np.ndarray  # (21, 3) hand A in its own frame
    other_points: np.ndarray   # (21, 3) hand B in hand A's frame
    translation: np.ndarray    # (3,) B's palm center in A's frame
    rotation: np.ndarray       # (3, 3) B's palm axes (as columns) in A's frame
    scale: float               # A's wrist -> middle MCP length, meters

    @property
    def normalized_translation(self) -> np.ndarray:
        """Translation in units of hand size, so it doesn't depend on hand size."""
        return self.translation / self.scale


def camera_matrix(w: int, h: int, dfov_deg: float = DFOV_DEG) -> np.ndarray:
    f = (np.hypot(w, h) / 2) / np.tan(np.radians(dfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], dtype=np.float64)


def hand_in_camera(landmarks, world_landmarks, w: int, h: int, K: np.ndarray) -> Optional[np.ndarray]:
    """Place one hand's metric world landmarks in the camera frame via PnP.

    Camera frame: +X right, +Y down, +Z forward, meters. Returns (21, 3) or None.
    """
    obj = np.array([(p.x, p.y, p.z) for p in world_landmarks], dtype=np.float64)
    img = np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return obj @ R.T + tvec.ravel()


def palm_frame(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (origin, R) for a hand; R's columns are the X, Y, Z axes."""
    origin = points[PALM].mean(axis=0)
    y = points[MIDDLE_MCP] - points[WRIST]
    y /= np.linalg.norm(y)
    x = points[PINKY_MCP] - points[INDEX_MCP]
    x -= x.dot(y) * y
    x /= np.linalg.norm(x)
    z = np.cross(x, y)
    return origin, np.column_stack((x, y, z))


def _pick_hands(results) -> Optional[Tuple[int, int]]:
    """Indices (anchor, other) with the right hand as anchor, or None."""
    if len(results.hand_landmarks) < 2:
        return None
    by_score = sorted(range(len(results.handedness)),
                      key=lambda i: results.handedness[i][0].score, reverse=True)
    a, b = by_score[:2]
    la = results.handedness[a][0].category_name
    lb = results.handedness[b][0].category_name
    if la != lb:
        return (a, b) if la == "Right" else (b, a)
    # Labels collide (common when hands overlap). On an un-mirrored frame the
    # user's right hand is on the image left.
    xa = results.hand_landmarks[a][WRIST].x
    xb = results.hand_landmarks[b][WRIST].x
    return (a, b) if xa < xb else (b, a)


def relative_pose(results, image_size: Tuple[int, int],
                  dfov_deg: float = DFOV_DEG) -> Optional[RelativePose]:
    """Convert HandLandmarker results into the right hand's frame.

    `results` is the output of HandLandmarker.detect_for_video on an un-mirrored
    frame; `image_size` is (width, height). Returns None unless two hands are found.
    """
    picked = _pick_hands(results)
    if picked is None:
        return None
    w, h = image_size
    K = camera_matrix(w, h, dfov_deg)
    pts = []
    for i in picked:
        p = hand_in_camera(results.hand_landmarks[i], results.hand_world_landmarks[i], w, h, K)
        if p is None:
            return None
        pts.append(p)
    a_cam, b_cam = pts

    origin_a, R_a = palm_frame(a_cam)
    origin_b, R_b = palm_frame(b_cam)
    to_a = lambda p: (p - origin_a) @ R_a  # camera frame -> A's frame

    return RelativePose(
        anchor_label=results.handedness[picked[0]][0].category_name,
        other_label=results.handedness[picked[1]][0].category_name,
        anchor_points=to_a(a_cam),
        other_points=to_a(b_cam),
        translation=to_a(origin_b),
        rotation=R_a.T @ R_b,
        scale=float(np.linalg.norm(a_cam[MIDDLE_MCP] - a_cam[WRIST])),
    )
