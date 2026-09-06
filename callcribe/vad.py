"""Нарезка непрерывного звука на законченные фразы по паузам."""

from __future__ import annotations

import queue
import threading
from collections import deque

import numpy as np
import webrtcvad

from .audio import AudioCapture
from .config import Config
from .models import Utterance


class VadSegmenter(threading.Thread):
    """Копит 30-мс кадры, отличает речь от тишины, на паузе отдаёт целую
    фразу (с прероллом) в очередь на распознавание.

    Отличия от наивной схемы:
      * метка времени — НАЧАЛО фразы, а не конец. Иначе 15-секундный монолог
        встанет в расшифровке позже короткой реплики, начавшейся после него;
      * хвостовая тишина обрезается до trailing_silence_keep_ms — длинный
        хвост тишины провоцирует галлюцинации whisper;
      * принудительная нарезка длинного монолога не сбрасывает состояние
        «идёт речь» и оставляет перехлёст, чтобы не разрезать слово пополам.
    """

    def __init__(
        self,
        capture: AudioCapture,
        cfg: Config,
        transcribe_queue: "queue.Queue[Utterance]",
        stop_event: threading.Event,
        pause_event: threading.Event,
    ):
        super().__init__(daemon=True, name=f"vad-{capture.label}")
        self.capture = capture
        self.cfg = cfg
        self.transcribe_queue = transcribe_queue
        self.stop_event = stop_event
        self.pause_event = pause_event

        self.vad = webrtcvad.Vad(cfg.vad_aggressiveness)
        self.frame_len = cfg.frame_len
        self.frame_dur = cfg.frame_ms / 1000

        self.preroll: deque[np.ndarray] = deque(maxlen=cfg.preroll_frames)
        self.buffer = np.zeros(0, dtype=np.float32)
        self.triggered = False
        self.voiced: list[np.ndarray] = []
        self.flags: list[bool] = []   # речь/тишина по кадрам, параллельно voiced
        self.silence_run = 0
        self.start_ts = 0.0
        self._buf_start_ts = 0.0

    # ------------------------------------------------------------------

    def _is_speech(self, frame: np.ndarray) -> bool:
        pcm16 = (np.clip(frame, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        return self.vad.is_speech(pcm16, self.cfg.sample_rate_target)

    def _flush(self, *, drop_frames: int = 0, keep_tail: int = 0, continued: bool = False) -> None:
        """Отправляет накопленное на распознавание.

        drop_frames — сколько кадров хвостовой тишины выбросить;
        keep_tail   — сколько кадров оставить началом следующей фразы;
        continued   — остаёмся ли в состоянии «идёт речь».
        """
        keep = max(0, len(self.voiced) - drop_frames)
        payload = self.voiced[:keep] if drop_frames else self.voiced
        if payload:
            audio = np.concatenate(payload)
            duration_ms = len(audio) / self.cfg.sample_rate_target * 1000
            if duration_ms >= self.cfg.min_utterance_ms:
                self.transcribe_queue.put(Utterance(self.capture.label, self.start_ts, audio))

        tail = self.voiced[-keep_tail:] if keep_tail else []
        emitted = len(self.voiced) - len(tail)
        self.voiced = list(tail)
        self.flags = self.flags[-keep_tail:] if keep_tail else []
        self.silence_run = 0
        self.triggered = continued

        if continued:
            self.start_ts += emitted * self.frame_dur
        else:
            self.preroll.clear()
            self.start_ts = 0.0

    def _find_split(self, min_gap: int) -> int | None:
        """Индекс, по которому лучше всего разрезать накопленную речь.

        Ищем самую длинную паузу в буфере и режем по её середине. Порог
        подбирать не нужно — берём лучшее из того, что реально было; если
        подходящей паузы нет вовсе, возвращаем None и режем по живому.
        """
        best_len, best_end, run = 0, -1, 0
        for index in range(self.cfg.min_chunk_frames, len(self.flags)):
            if self.flags[index]:
                run = 0
            else:
                run += 1
                if run > best_len:
                    best_len, best_end = run, index
        if best_len < min_gap:
            return None
        cut = best_end - best_len // 2
        return cut if cut > self.cfg.min_chunk_frames else None

    def _split_at(self, cut: int) -> None:
        """Отдаёт voiced[:cut] на распознавание, остальное продолжает копиться."""
        rest = len(self.voiced) - cut
        self._flush(drop_frames=rest, keep_tail=rest, continued=True)

    def _reset(self) -> None:
        """Сброс состояния — например, при паузе, чтобы не склеить фразы
        по разные стороны от неё."""
        if self.triggered:
            self._flush(drop_frames=max(0, self.silence_run - self.cfg.keep_silence_frames))
        self.buffer = np.zeros(0, dtype=np.float32)
        self.preroll.clear()

    # ------------------------------------------------------------------

    def run(self) -> None:
        was_paused = False

        while True:
            try:
                chunk_ts, chunk = self.capture.out_queue.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                if self.pause_event.is_set() and not was_paused:
                    self._reset()
                    was_paused = True
                continue

            if was_paused:
                was_paused = False

            # chunk_ts проставлен потоком захвата и означает конец блока,
            # отсюда получаем время начала всего, что лежит в буфере.
            self.buffer = np.concatenate([self.buffer, chunk])
            self._buf_start_ts = chunk_ts - len(self.buffer) / self.cfg.sample_rate_target

            while len(self.buffer) >= self.frame_len:
                frame_ts = self._buf_start_ts
                frame = self.buffer[: self.frame_len]
                self.buffer = self.buffer[self.frame_len :]
                self._buf_start_ts += self.frame_dur

                speech = self._is_speech(frame)

                if not self.triggered:
                    self.preroll.append(frame)
                    if speech:
                        self.triggered = True
                        self.voiced = list(self.preroll)
                        # Преролл — это тишина перед речью, речевой в нём
                        # только последний кадр. flags обязан идти ровно
                        # параллельно voiced, иначе разрез уедет по индексам.
                        self.flags = [False] * (len(self.preroll) - 1) + [True]
                        # Преролл заканчивается текущим кадром, значит фраза
                        # началась на (len(preroll) - 1) кадров раньше него.
                        self.start_ts = frame_ts - (len(self.preroll) - 1) * self.frame_dur
                        self.preroll.clear()
                        self.silence_run = 0
                    continue

                self.voiced.append(frame)
                self.flags.append(speech)
                self.silence_run = 0 if speech else self.silence_run + 1

                if self.silence_run >= self.cfg.end_silence_frames:
                    # Фраза закончилась.
                    self._flush(
                        drop_frames=max(0, self.silence_run - self.cfg.keep_silence_frames)
                    )
                    continue

                # min(), а не soft_frames: жёсткий предел обязан срабатывать
                # даже если мягкий выкрутили выше него.
                if len(self.voiced) < min(self.cfg.soft_frames, self.cfg.max_frames):
                    continue

                # Речь идёт давно. Режем по самой заметной паузе в буфере —
                # так стык приходится на промежуток между словами, а не на
                # середину слова, и текст появляется, не дожидаясь конца
                # монолога.
                cut = self._find_split(self.cfg.split_silence_frames)
                if cut is not None:
                    self._split_at(cut)
                elif len(self.voiced) >= self.cfg.max_frames:
                    # Ни одной паузы за всё время — человек читает список
                    # без единого вдоха. Режем по любой тишине, а если нет
                    # и её, то по живому с перехлёстом.
                    cut = self._find_split(1)
                    if cut is not None:
                        self._split_at(cut)
                    else:
                        self._flush(keep_tail=self.cfg.overlap_frames, continued=True)

        # Дослать то, что осталось на момент остановки.
        if self.triggered:
            self._flush(drop_frames=max(0, self.silence_run - self.cfg.keep_silence_frames))
