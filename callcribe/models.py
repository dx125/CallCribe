"""Типы, которые ходят между потоками."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Utterance:
    """Законченная фраза, вырезанная VAD-ом и ждущая распознавания."""

    label: str          # "Я" / "Собеседник"
    ts: float           # wall-clock НАЧАЛА фразы, не конца
    audio: np.ndarray   # float32 mono 16 кГц

    @property
    def duration(self) -> float:
        return len(self.audio) / 16_000


@dataclass(slots=True)
class Line:
    """Распознанная строка, готовая к выводу и сохранению."""

    ts: float
    label: str
    text: str
