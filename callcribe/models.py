"""Типы, которые ходят между потоками."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Utterance:
    """Законченная фраза, вырезанная VAD-ом и ждущая распознавания."""

    label: str          # "Я" / "Собеседник"
    ts: float           # wall-clock НАЧАЛА фразы, не конца
    audio: np.ndarray   # float32 mono, частота — в sample_rate
    sample_rate: int = 16_000
    """Частота звука в audio.

    Едет вместе с массивом, а не берётся числом из duration: в массиве
    частоты нет, а sample_rate_target настраивается (config.py). С зашитым
    16 000 смена частоты молча портила бы duration — и вместе с ней
    предупреждение «распознавание медленнее реального времени», то есть
    ровно тот показатель, по которому эту настройку и крутят.
    """

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate


@dataclass(slots=True)
class Line:
    """Распознанная строка, готовая к выводу и сохранению."""

    ts: float
    label: str
    text: str
