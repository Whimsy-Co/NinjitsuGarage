from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from src.features import N_FEATURES, relative_pose_features
from src.hands import RelativePose

MODEL_PATH = Path("models/gesture_mlp.pt")


class GestureMLP(nn.Module):
    """Small MLP over standardized pose features; outputs one logit per gesture."""

    def __init__(self, n_classes: int, hidden: Tuple[int, ...] = (128, 64), dropout: float = 0.2):
        super().__init__()
        layers, width = [], N_FEATURES
        for h in hidden:
            layers += [nn.Linear(width, h), nn.ReLU(), nn.Dropout(dropout)]
            width = h
        layers.append(nn.Linear(width, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def save_checkpoint(path: Path, model: GestureMLP, labels: List[str], hidden, mean, std):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "labels": list(labels),
        "hidden": tuple(hidden),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
    }, path)


class GestureClassifier:
    """Loads a trained GestureMLP and classifies live RelativePose frames."""

    def __init__(self, path: Path = MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.labels: List[str] = ckpt["labels"]
        self.mean = ckpt["mean"]
        self.std = ckpt["std"]
        self.model = GestureMLP(len(self.labels), ckpt["hidden"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Class probabilities for features of shape (N_FEATURES,) or (S, N_FEATURES)."""
        x = torch.from_numpy((np.atleast_2d(features) - self.mean) / self.std)
        p = torch.softmax(self.model(x), dim=1).numpy()
        return p[0] if features.ndim == 1 else p

    def predict(self, rp: RelativePose) -> Tuple[str, float]:
        """Most likely gesture label for one frame, and its probability."""
        p = self.predict_proba(relative_pose_features(rp))
        i = int(p.argmax())
        return self.labels[i], float(p[i])
