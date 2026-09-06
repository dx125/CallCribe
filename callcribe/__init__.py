"""CallCribe — локальная расшифровка звонков в реальном времени (Windows).

Микрофон + системный звук (WASAPI loopback) -> VAD-сегментация по паузам ->
faster-whisper -> живое окно с текстом. Ничего никуда не отправляется.
"""

import sys

__version__ = "1.0.0"

# Проверка идёт до любых импортов пакета и намеренно обходится без i18n:
# на слишком старом Python сам разбор модулей и падает, так что перевести
# это сообщение уже нечем. Нижняя граница — 3.10: dataclass(slots=True).
#
# Без этой проверки человек получает TypeError из глубины dataclasses и
# идёт чинить приложение вместо того, чтобы обновить Python.
MIN_PYTHON = (3, 10)

if sys.version_info < MIN_PYTHON:
    _need = ".".join(str(part) for part in MIN_PYTHON)
    _have = ".".join(str(part) for part in sys.version_info[:3])
    raise SystemExit(
        f"CallCribe needs Python {_need} or newer, but this is {_have}.\n"
        f"CallCribe требуется Python {_need} или новее, а здесь {_have}.\n"
        f"  {sys.executable}"
    )
