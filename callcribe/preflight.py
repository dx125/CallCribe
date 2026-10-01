"""Проверка, что модель в этом окружении вообще грузится.

Сломанный ctranslate2 не бросает исключение — он уносит процесс целиком
(на Windows это 0xC0000005, в оболочке видно как segfault), мимо любого
except и любого finally. Изнутри поймать это нечем: after-логики у
падения нет. Поэтому загрузку пробуем ОТДЕЛЬНЫМ процессом и смотрим на
код возврата — снаружи видно и падение, и обычную ошибку.

Замерено на живой поломке: pip, которому в requirements.txt не задали
верхнюю границу, собрал ctranslate2 4.8.2 — и тот падал на загрузке
ЛЮБОЙ модели, включая tiny, на CPU и на CUDA одинаково, не успев
напечатать ни строки даже с CT2_VERBOSE=3. Приложение при этом просто
исчезало: под pythonw.exe окна с ошибкой нет, консоли нет, в файле
настроек тоже ничего. selftest при этом показывал 180 проверок из 180 —
загрузку модели он не пробовал, а попробовав в своём процессе, умер бы
вместе с ней.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Код возврата 1 питон отдаёт на обычном исключении, 0 — на успехе.
# Всё остальное — падение среды исполнения, а не ошибка модели.
_EXIT_OK = 0
_EXIT_EXCEPTION = 1

_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
from callcribe.cuda import prepare_cuda_dll_path
prepare_cuda_dll_path()
from faster_whisper import WhisperModel
WhisperModel(sys.argv[2], device=sys.argv[3], compute_type=sys.argv[4],
             local_files_only=True)
print("ok")
"""


_VAD_PROBE = """
import sys
sys.path.insert(0, sys.argv[1])
import numpy as np
from faster_whisper.vad import VadOptions, get_speech_timestamps
get_speech_timestamps(np.zeros(16000, dtype=np.float32), VadOptions())
print("ok")
"""


def vad_filter_works(timeout: float = 120) -> tuple[bool, str]:
    """Работает ли фильтр Silero VAD. Тоже отдельным процессом, и тоже
    не от перестраховки.

    Фильтр живёт внутри faster-whisper, но считает его onnxruntime —
    вторая нативная библиотека рядом с ctranslate2. Замерено на живой
    машине: модель грузится, «Слушаю» в окне, а на ПЕРВОЙ же фразе
    процесс исчезает — ровно на инициализации Silero. Без фильтра та же
    фраза разбирается как надо.

    Проба дешёвая: секунда тишины через сам фильтр, без whisper. Гоняем
    один раз на старте, а дальше просто не включаем фильтр, если он здесь
    не жилец — на входе и так стоит webrtcvad.
    """
    root = str(Path(__file__).resolve().parents[1])
    try:
        done = subprocess.run(
            [sys.executable, "-c", _VAD_PROBE, root],
            capture_output=True, text=True, timeout=timeout,
            creationflags=_no_console(),
        )
    except subprocess.TimeoutExpired:
        return False, f"проба не уложилась в {timeout:.0f} с"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if done.returncode == _EXIT_OK:
        return True, ""
    if done.returncode == _EXIT_EXCEPTION:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        return False, lines[-1] if lines else "проба не удалась без объяснений"
    return False, f"проба упала, код возврата {done.returncode}"


def _hub_cache() -> Path:
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def unfinished_downloads() -> list[Path]:
    """Огрызки оборванных загрузок в кэше Hugging Face.

    Оборванная загрузка оставляет файл .incomplete, и следующая попытка
    его НЕ продолжает — она заводит свой, с новым случайным именем.
    Замерено на живой машине: восемь таких огрызков на 2.2 ГБ и загрузка,
    которая раз за разом умирала посреди файла; сразу после их удаления
    та же модель скачалась с первой попытки. Поэтому при неудачной
    загрузке о них надо сказать — сами они не рассосутся.
    """
    cache = _hub_cache()
    if not cache.is_dir():
        return []
    try:
        return sorted(cache.rglob("*.incomplete"))
    except OSError:
        return []


def _no_console() -> int:
    """Не мигать окном консоли: приложение живёт под pythonw.exe."""
    return getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0


def model_loads(
    name: str,
    device: str = "cpu",
    compute: str = "int8",
    timeout: float = 600,
) -> tuple[bool, str]:
    """Грузится ли модель. Возвращает (получилось, чем объяснить).

    Модель обязана быть уже в кэше: проба ходит с local_files_only=True,
    чтобы не качать гигабайты ради проверки и не зависеть от сети.
    """
    root = str(Path(__file__).resolve().parents[1])
    try:
        done = subprocess.run(
            [sys.executable, "-c", _PROBE, root, name, device, compute],
            capture_output=True, text=True, timeout=timeout,
            creationflags=_no_console(),
        )
    except subprocess.TimeoutExpired:
        return False, f"загрузка не уложилась в {timeout:.0f} с"
    except OSError as exc:
        return False, f"{type(exc).__name__}: {exc}"

    if done.returncode == _EXIT_OK:
        return True, ""

    # Осмысленная ошибка: модели нет в кэше, папка не та, формат не тот.
    # Последняя строка трассировки — как раз она.
    if done.returncode == _EXIT_EXCEPTION:
        lines = [line for line in done.stderr.splitlines() if line.strip()]
        return False, lines[-1] if lines else "загрузка не удалась без объяснений"

    # Всё остальное — упавшая среда. Код возврата здесь и есть диагноз:
    # 0xC0000005 (то же число как знаковое: -1073741819) — access violation.
    return False, f"процесс загрузки упал, код возврата {done.returncode}"
