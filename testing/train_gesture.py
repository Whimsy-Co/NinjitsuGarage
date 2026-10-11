"""
Train the gesture MLP on recordings from testing/build_gesture.py.

Every data/gestures/*.npz is loaded and its `name` is used as the label (record
a "None" gesture of idle or random hands so the model can say "no gesture").

Frames within one hold are nearly identical, so a random train/test split would
overstate accuracy. Instead it evaluates by holding out one angle at a time
(head-on, turned left, turned right) and training on the other two. Then it
trains on everything and saves models/gesture_mlp.pt.

Usage: python testing/train_gesture.py [--epochs 60]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import pose_features 
from src.gesture_model import MODEL_PATH, GestureMLP, save_checkpoint 

DATA_DIR = Path("data/gestures")
HIDDEN = (128, 64)
ANGLES = ["head-on", "turned left", "turned right"] # 3 distances each
NOISE = 0.05  # gaussian noise added to standardized features during training


def load_dataset():
    X, y, angle, labels = [], [], [], []
    for f in sorted(DATA_DIR.glob("*.npz")):
        d = np.load(f)
        if "view_index" not in d.files:
            print(f"Skipping {f.name}: old format")
            continue
        name = str(d["name"])
        if name not in labels:
            labels.append(name)
        X.append(pose_features(d["anchor_points"], d["other_points"], d["translation"],
                               d["normalized_translation"], d["rotation"], d["scale"]))
        y.append(np.full(len(d["view_index"]), labels.index(name)))
        angle.append(d["view_index"] // 3)
    return np.concatenate(X), np.concatenate(y), np.concatenate(angle), labels


def train(X, y, n_classes, epochs, seed=0):
    """Returns (model, mean, std) trained on X, y."""
    torch.manual_seed(seed)
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - mean) / std)
    yt = torch.from_numpy(y).long()
    model = GestureMLP(n_classes, HIDDEN)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(Xt)).split(256):
            xb = Xt[idx] + NOISE * torch.randn_like(Xt[idx])
            opt.zero_grad()
            loss_fn(model(xb), yt[idx]).backward()
            opt.step()
    model.eval()
    return model, mean, std


@torch.no_grad()
def predict(model, mean, std, X):
    return model(torch.from_numpy((X - mean) / std)).argmax(1).numpy()


def print_confusion(cm, labels):
    w = max(len(l) for l in labels)
    print(" " * (w + 2) + " ".join(f"{i:>4d}" for i in range(len(labels))))
    for i, row in enumerate(cm):
        acc = row[i] / max(row.sum(), 1)
        print(f"{labels[i]:>{w}} {i:>2d}" + "".join(f"{v:>5d}" for v in row) + f"   {acc:.0%}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--epochs", type=int, default=60)
    args = p.parse_args()

    X, y, angle, labels = load_dataset()
    print(f"{len(X)} samples, {len(labels)} gestures: {', '.join(labels)}\n")

    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for a, name in enumerate(ANGLES):
        test = angle == a
        model, mean, std = train(X[~test], y[~test], len(labels), args.epochs)
        pred = predict(model, mean, std, X[test])
        np.add.at(cm, (y[test], pred), 1)
        print(f"Held out {name:<13} accuracy {np.mean(pred == y[test]):.1%}")
    print(f"\nOverall held-out accuracy {np.trace(cm) / cm.sum():.1%}")
    print("Confusion matrix (rows = true, columns = predicted):")
    print_confusion(cm, labels)

    model, mean, std = train(X, y, len(labels), args.epochs)
    save_checkpoint(MODEL_PATH, model, labels, HIDDEN, mean, std)
    print(f"\nTrained on all data, saved {MODEL_PATH}")


if __name__ == "__main__":
    main()
