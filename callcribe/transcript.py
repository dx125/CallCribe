"""Сохранение расшифровки.

Пишем построчно по мере распознавания — падение, BSOD или kill процесса
не должны съедать сорокаминутный звонок. При штатном закрытии файл
переписывается заново, отсортированный по времени начала фраз: два
независимых канала приходят вперемешку.

Побочная польза этой пересборки: метки говорящих и заголовок берутся из
языка интерфейса в момент записи, и если его переключили посреди звонка,
итоговый файл всё равно выходит на одном языке — на том, что выбран в
конце.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from .config import Config
from .i18n import speaker, t
from .models import Line


class TranscriptWriter:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.lines: list[Line] = []
        self.path: Path | None = None
        self._fh = None
        self._closed = False
        self._lock = threading.Lock()

    # ------------------------------------------------------------------

    def _format(self, line: Line) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(line.ts))
        prefix = f"**{speaker(line.label)}** " if self.cfg.show_speaker_labels else ""
        return f"`{stamp}` {prefix}{line.text}"

    def _header(self, first_ts: float) -> str:
        started = time.strftime("%Y-%m-%d %H:%M", time.localtime(first_ts))
        return t("transcript.header", started=started) + "\n\n"

    def _open(self, first_ts: float) -> None:
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(first_ts))
        self.path = self.cfg.output_dir / f"call_{stamp}.md"
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(self._header(first_ts))
        self._fh.flush()

    # ------------------------------------------------------------------

    def append(self, line: Line) -> None:
        """Вызывается из потока распознавания."""
        with self._lock:
            if self._closed:
                # finalize() уже отработал (поток распознавания не уложился
                # в таймаут join). Открывать второй файл нельзя — строка
                # просто теряется, всё остальное уже на диске.
                return
            if self._fh is None:
                self._open(line.ts)
            self.lines.append(line)
            self._fh.write(self._format(line) + "\n\n")
            self._fh.flush()

    def finalize(self) -> Path | None:
        """Пересобирает файл в хронологическом порядке. Возвращает путь."""
        with self._lock:
            self._closed = True
            if self._fh is not None:
                self._fh.close()
                self._fh = None
            if not self.lines or self.path is None:
                return None

            ordered = sorted(self.lines, key=lambda line: line.ts)
            body = "\n\n".join(self._format(line) for line in ordered)
            self.path.write_text(self._header(ordered[0].ts) + body + "\n", encoding="utf-8")
            return self.path
