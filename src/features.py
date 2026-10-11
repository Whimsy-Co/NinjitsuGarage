import numpy as np

from src.hands import RelativePose

# 63 right-hand shape + 63 left-hand shape + 3 translation + 6 rotation
N_FEATURES = 135


def pose_features(anchor_points, other_points, translation, normalized_translation,
                  rotation, scale) -> np.ndarray:
    """Feature vectors for a batch of two-hand poses, shape (S, N_FEATURES).

    Inputs are the arrays saved by testing/build_gesture.py, each with a leading
    sample dimension. Both hand shapes are scaled by the right hand's size, and the
    left hand's shape is taken in its own palm frame so it carries no position.
    """
    s = np.asarray(scale, dtype=np.float64).reshape(-1, 1, 1)
    anchor = anchor_points / s
    # Undo B's placement in A's frame: (p - t) @ (R_A^T R_B) = B's own palm frame.
    other = np.einsum("sij,sjk->sik", other_points - translation[:, None, :], rotation) / s
    rot6 = rotation[:, :, :2]  # first two axes; the third is their cross product
    return np.concatenate([
        anchor.reshape(len(s), -1),
        other.reshape(len(s), -1),
        normalized_translation,
        rot6.reshape(len(s), -1),
    ], axis=1).astype(np.float32)


def relative_pose_features(rp: RelativePose) -> np.ndarray:
    """Feature vector for one live RelativePose, shape (N_FEATURES,)."""
    return pose_features(rp.anchor_points[None], rp.other_points[None], rp.translation[None],
                         rp.normalized_translation[None], rp.rotation[None],
                         np.array([rp.scale]))[0]
