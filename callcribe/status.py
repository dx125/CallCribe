"""Канал сообщений из рабочих потоков в GUI.

Без него исключение в фоновом потоке (не загрузилась модель, отвалилось
устройство) умирает молча, а окно вечно висит с надписью
"Загружаю модель...".
"""

from __future__ import annotations

import logging
import queue
import sys
from dataclasses import dataclass

INFO = "info"
WARN = "warn"
FATAL = "fatal"

# Всё, что увидел пользователь, должно остаться и в журнале: окно
# показывает сообщение секунды четыре, а разбираться в случившемся
# приходится потом — по файлу, см. diagnostics.py.
_LOG = logging.getLogger("callcribe")
_LOG_LEVELS = {INFO: logging.INFO, WARN: logging.WARNING, FATAL: logging.ERROR}


@dataclass(slots=True)
class Notice:
    level: str
    text: str


class Notifier:
    """Потокобезопасный кран сообщений. GUI опрашивает .queue."""

    def __init__(self) -> None:
        self.queue: "queue.Queue[Notice]" = queue.Queue()

    def _put(self, level: str, text: str) -> None:
        # Очередь заполняется первой и всегда: именно из неё сообщение
        # попадает в окно. Печать в консоль — удобство, и она вполне может
        # не удаться (перенаправленный вывод, закрытый канал). Терять из-за
        # этого сообщение, тем более fatal, нельзя.
        self.queue.put(Notice(level, text))
        _LOG.log(_LOG_LEVELS.get(level, logging.INFO), text)
        stream = sys.stderr if level in (WARN, FATAL) else sys.stdout
        # Под pythonw потоки подменены журналом (diagnostics.LogStream), и
        # печать туда записала бы ту же строку второй раз.
        if stream is None or getattr(stream, "is_log_stream", False):
            return
        try:
            print(f"[{level}] {text}", file=stream, flush=True)
        except (UnicodeEncodeError, OSError, ValueError):
            pass

    def info(self, text: str) -> None:
        self._put(INFO, text)

    def warn(self, text: str) -> None:
        self._put(WARN, text)

    def fatal(self, text: str) -> None:
        self._put(FATAL, text)


def use_utf8_console() -> None:
    """Разрешить кириллицу в stdout/stderr.

    В настоящей консоли Python пишет через WriteConsoleW и всё в порядке.
    Но стоит перенаправить вывод в файл или канал (`run.cmd > log.txt`,
    запуск из IDE), как берётся кодировка локали — cp1252, — и первая же
    строка с русским текстом валит процесс UnicodeEncodeError'ом ещё до
    появления окна. Под pythonw потоков нет вовсе, там печатать некуда.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass
