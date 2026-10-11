from collections import deque
from typing import List, Optional

import numpy as np


class SequenceMatcher:

    def __init__(self, sequence: List[str], labels: List[str], hold_s: float = 0.5,
                 step_timeout_s: float = 4.0, window: int = 10, min_prob: float = 0.8,
                 idle_label: str = "None"):
        missing = [m for m in sequence if m not in labels]
        if missing:
            raise ValueError(f"Moves not in the model's labels: {missing}")
        self.sequence = list(sequence)
        self.labels = list(labels)
        self.hold_s = hold_s
        self.step_timeout_s = step_timeout_s
        self.min_prob = min_prob
        self.idle_label = idle_label
        self._history = deque(maxlen=window)
        self.reset()

    def reset(self):
        self.step = 0 
        self._last_step_time = None  
        self._candidate = None       
        self._candidate_since = None
        self._history.clear()

    @property
    def expected(self) -> str:
        return self.sequence[self.step]

    def smoothed(self) -> Optional[str]:
        """Only returns a confident guess"""
        if len(self._history) < self._history.maxlen:
            return None
        votes = np.bincount([int(p.argmax()) for p in self._history], minlength=len(self.labels))
        best = int(votes.argmax())
        if np.mean([p[best] for p in self._history]) < self.min_prob:
            return None
        return self.labels[best]

    def update(self, probs: Optional[np.ndarray], now: float) -> Optional[str]:
        """Feed one frame (probs is None when two hands weren't found).

        complete if the sequence is done, step if you got the current one right,
        or reset if you got it wrong (which clears progress).
        """
        if self.step > 0 and now - self._last_step_time > self.step_timeout_s:
            self.reset()
            return "reset"
        if probs is None:
            self._history.clear()
            self._candidate = None
            return None

        self._history.append(probs)
        label = self.smoothed()
        if label != self._candidate:
            self._candidate, self._candidate_since = label, now
        if label is None or now - self._candidate_since < self.hold_s:
            return None

        previous = self.sequence[self.step - 1] if self.step > 0 else None
        if label == self.expected:
            self.step += 1
            self._last_step_time = now
            self._candidate = None  # make the next move earn its own hold
            self._history.clear()
            if self.step == len(self.sequence):
                self.reset()
                return "complete"
            return "step"
        if label in (previous, self.idle_label):
            return None
        if self.step > 0:
            self.reset()
            return "reset"
        return None
