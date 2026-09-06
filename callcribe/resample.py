"""Ресемплинг в 16 кГц с антиалиасингом и состоянием между чанками.

Зачем не голый np.interp: устройства отдают 44.1/48 кГц, целевая частота
16 кГц. Прореживание 48 -> 16 без ФНЧ заворачивает всё, что выше 8 кГц,
обратно в речевую полосу — заметнее всего на шипящих и на шуме. Плюс
независимый np.interp на каждом 100-мс блоке даёт разрыв фазы на стыках.

Основной путь — libsoxr (потоковый режим, состояние внутри). Фолбэк —
numpy: windowed-sinc ФНЧ с переносом хвоста между вызовами и дробная
интерполяция с переносом позиции.
"""

from __future__ import annotations

import numpy as np

try:
    import soxr
except ImportError:  # pragma: no cover - зависит от окружения
    soxr = None

HAVE_SOXR = soxr is not None

_EMPTY = np.zeros(0, dtype=np.float32)


class _Passthrough:
    def process(self, x: np.ndarray) -> np.ndarray:
        return x.astype(np.float32, copy=False)


class _SoxrResampler:
    def __init__(self, orig_sr: int, target_sr: int):
        self._stream = soxr.ResampleStream(
            orig_sr, target_sr, 1, dtype="float32", quality="HQ"
        )

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return _EMPTY
        return self._stream.resample_chunk(x.astype(np.float32, copy=False))


class _NumpyResampler:
    """Фолбэк без внешних зависимостей. Хуже soxr, но без алиасинга."""

    def __init__(self, orig_sr: int, target_sr: int, num_taps: int = 101):
        self.step = orig_sr / target_sr
        self.taps: np.ndarray | None = None
        self._tail = _EMPTY

        if orig_sr > target_sr:
            # Срез на 0.45 от целевой Найквиста, нормированный к исходной fs.
            fc = 0.45 * target_sr / orig_sr
            n = np.arange(num_taps)
            h = np.sinc(2 * fc * (n - (num_taps - 1) / 2)) * np.hamming(num_taps)
            self.taps = (h / h.sum()).astype(np.float32)
            self._tail = np.zeros(num_taps - 1, dtype=np.float32)

        self._buf = _EMPTY
        self._pos = 0.0  # дробная позиция чтения внутри _buf

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return _EMPTY
        x = x.astype(np.float32, copy=False)

        if self.taps is not None:
            padded = np.concatenate([self._tail, x])
            self._tail = padded[-(len(self.taps) - 1) :]
            x = np.convolve(padded, self.taps, mode="valid").astype(np.float32)

        self._buf = np.concatenate([self._buf, x])
        if len(self._buf) < 2:
            return _EMPTY

        span = len(self._buf) - 1 - self._pos
        if span < 0:
            return _EMPTY
        n_out = int(np.floor(span / self.step)) + 1
        if n_out <= 0:
            return _EMPTY

        idx = self._pos + self.step * np.arange(n_out, dtype=np.float64)
        out = np.interp(idx, np.arange(len(self._buf)), self._buf).astype(np.float32)

        # Сдвигаем буфер, перенося дробный остаток позиции в следующий вызов.
        consumed = int(np.floor(idx[-1]))
        self._buf = self._buf[consumed:]
        self._pos = idx[-1] - consumed + self.step
        return out


def make_resampler(orig_sr: int, target_sr: int):
    """Создаёт потоковый ресемплер. Состояние живёт внутри объекта —
    на каждый источник звука нужен свой экземпляр."""
    if orig_sr == target_sr:
        return _Passthrough()
    if HAVE_SOXR:
        return _SoxrResampler(orig_sr, target_sr)
    return _NumpyResampler(orig_sr, target_sr)
