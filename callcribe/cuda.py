"""Подготовка путей к CUDA-библиотекам перед импортом ctranslate2.

CTranslate2 (движок faster-whisper) на Windows ищет cuBLAS и cuDNN 9 в
PATH. Колёса nvidia-cublas-cu12 / nvidia-cudnn-cu12 кладут DLL в
site-packages/nvidia/*/bin, куда система сама не смотрит. Отсюда две самые
частые ошибки: "Could not locate cudnn_ops64_9.dll" и
"Library cublas64_12.dll is not found or cannot be loaded".

Каталоги регистрируются ДВУМЯ способами, и оба нужны:

  * os.add_dll_directory — работает только для загрузки с флагами
    LOAD_LIBRARY_SEARCH_*, то есть для питоновских расширений;
  * os.environ["PATH"] — CTranslate2 подгружает cuBLAS лениво, уже из C++,
    обычным LoadLibrary. Тот идёт по устаревшему порядку поиска, где
    add_dll_directory не участвует, зато участвует PATH. Без этого модель
    загрузится на cuda успешно, а упадёт только на первой же фразе.

prepare_cuda_dll_path() должна вызываться ДО импорта ctranslate2 /
faster_whisper.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_prepared = False
_handles: list = []  # держим ссылки, иначе каталоги отвалятся


def _nvidia_bin_dirs() -> list[Path]:
    dirs: list[Path] = []
    for entry in sys.path:
        nvidia = Path(entry) / "nvidia"
        if not nvidia.is_dir():
            continue
        for package in sorted(nvidia.iterdir()):
            for sub in ("bin", "lib"):
                candidate = package / sub
                if candidate.is_dir() and any(candidate.glob("*.dll")):
                    dirs.append(candidate)
    return dirs


_found: list[str] = []


def prepare_cuda_dll_path() -> list[str]:
    """Регистрирует каталоги с CUDA-DLL. Возвращает то, что нашлось."""
    global _prepared
    if _prepared:
        return list(_found)

    if sys.platform == "win32":
        for directory in _nvidia_bin_dirs():
            path = str(directory)
            try:
                _handles.append(os.add_dll_directory(path))
            except OSError:
                pass
            _found.append(path)

        if _found:
            # Именно отсюда cuBLAS находится при ленивой подгрузке из C++.
            current = os.environ.get("PATH", "")
            missing = [p for p in _found if p not in current.split(os.pathsep)]
            if missing:
                os.environ["PATH"] = os.pathsep.join([*missing, current])

    _prepared = True
    return list(_found)


def cuda_device_count() -> int:
    """Число видимых CUDA-устройств. 0, если CUDA недоступна или сломана."""
    prepare_cuda_dll_path()
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


# Библиотеки, которые CTranslate2 подгружает уже во время счёта, а не при
# загрузке модели. Внутри группы достаточно любой: имя содержит мажорную
# версию, и при переезде на следующую CUDA соседнее имя избавит от ложного
# отказа от видеокарты.
_COMPUTE_DLLS: tuple[tuple[str, ...], ...] = (
    ("cublas64_12.dll", "cublas64_13.dll"),
    ("cudnn_ops64_9.dll", "cudnn_ops64_10.dll"),
)


def missing_cuda_libraries() -> list[str]:
    """Библиотеки счёта, которых не хватает. Пустой список — всё на месте.

    Это НЕ то же самое, что cuda_device_count(). Устройство видно через
    nvcuda.dll, а он ставится вместе с драйвером и есть на любой машине с
    видеокартой. Считать же CTranslate2 будет через cuBLAS и cuDNN, а они
    приезжают отдельными колёсами (requirements-gpu.txt).

    Разойтись эти два факта могут запросто: поставили requirements.txt на
    машине с видеокартой — устройство видно, библиотек нет. Модель при
    этом загружается на cuda без единой жалобы, в окне бодро написано
    "cuda/float16", и падает КАЖДАЯ фраза:
    "Library cublas64_12.dll is not found or cannot be loaded".

    Поэтому проверяем до выбора устройства, а не после первой потерянной
    фразы. Ищем тем же способом, каким будет искать CTranslate2 (обычный
    LoadLibrary по путям), так что системная установка CUDA тоже найдётся.
    """
    if sys.platform != "win32":
        return []

    prepare_cuda_dll_path()
    missing: list[str] = []
    for group in _COMPUTE_DLLS:
        for name in group:
            try:
                ctypes.WinDLL(name)
                break
            except OSError:
                continue
        else:
            missing.append(group[0])
    return missing
