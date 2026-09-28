"""Сборка приложения и главный цикл."""

from __future__ import annotations

import dataclasses
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

import pyaudiowpatch as pyaudio

from .asr import TranscriberWorker
from .audio import AudioCapture, describe, get_loopback_devices, get_mic_device
from .config import CFG, Config, LanguageSetting, ModelSetting, language_name
from .i18n import set_language, t, ui_language_name
from .models import Line, Utterance
from .resample import HAVE_SOXR
from .settings import SettingsStore, model_display
from .status import Notifier
from .transcript import TranscriptWriter
from .ui import TranscriptWindow
from .vad import VadSegmenter

# Метки каналов — ключи, а не готовые подписи: они попадают и в окно, и в
# файл, а язык интерфейса меняется на ходу. Перевод берётся в момент
# показа, см. i18n.speaker().
MIC_LABEL = "me"
LOOPBACK_LABEL = "them"


def _show_startup_error(message: str) -> None:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(t("app.audio_error_title"), message)
    root.destroy()


def _pick_devices(notifier: Notifier) -> tuple[dict | None, list[dict]]:
    """Хотя бы один вывод обязателен, микрофон — нет.

    Без микрофона запись половины разговора всё равно полезнее, чем отказ
    стартовать: гарнитуру часто подключают уже после запуска.
    """
    pa = pyaudio.PyAudio()
    try:
        loopbacks = get_loopback_devices(pa)   # без них смысла нет — пусть падает
        try:
            mic = get_mic_device(pa)
        except Exception as exc:
            notifier.warn(t("app.mic_unavailable", error=exc))
            mic = None
        return mic, loopbacks
    finally:
        pa.terminate()


def load_settings(cfg: Config, store: SettingsStore | None = None):
    """Прочитать сохранённый выбор и наложить его на конфигурацию.

    Возвращает (store, cfg, проблемы). Язык интерфейса выставляется здесь
    же и первым делом: жалобы на этот же файл должны выйти уже на нём.
    """
    store = store if store is not None else SettingsStore()
    problems = store.load()
    set_language(store.data.ui_language)
    cfg = dataclasses.replace(
        cfg,
        ui_language=store.data.ui_language,
        language=store.data.language,
        whisper_model=store.data.whisper_model,
    )
    return store, cfg, problems


def run(
    cfg: Config | None = None,
    store: SettingsStore | None = None,
    problems: list[tuple[str, dict]] | None = None,
) -> None:
    cfg = cfg or CFG
    notifier = Notifier()

    # Настройки могли быть прочитаны раньше — в __main__, чтобы поверх них
    # легли ключи командной строки. Второй раз файл не читаем.
    problems = list(problems or ())
    if store is None:
        store, cfg, problems = load_settings(cfg)
    else:
        set_language(cfg.ui_language)

    for key, params in problems:
        notifier.warn(t(key, **params))
    for warning in cfg.validate():
        notifier.warn(warning)
    if not HAVE_SOXR:
        notifier.warn(t("app.no_soxr"))

    try:
        mic_device, loop_devices = _pick_devices(notifier)
    except Exception as exc:
        _show_startup_error(str(exc))
        sys.exit(1)

    notifier.info(t("app.loopback", device=describe(loop_devices[0])))
    for extra in loop_devices[1:]:
        notifier.info(t("app.loopback_extra", device=describe(extra)))
    if mic_device is not None:
        notifier.info(t("app.mic", device=describe(mic_device)))
    notifier.info(t("app.interface_language", language=ui_language_name(cfg.ui_language)))
    notifier.info(t("app.speech_language", language=language_name(cfg.language)))
    notifier.info(t("app.model", model=model_display(cfg.whisper_model)))

    # Настройки, живущие отдельно от Config: их меняют из окна прямо во
    # время звонка, а применяет поток распознавания.
    language = LanguageSetting(cfg.language)
    model = ModelSetting(cfg.whisper_model)

    stop_event = threading.Event()
    pause_event = threading.Event()
    transcribe_q: "queue.Queue[Utterance]" = queue.Queue()
    gui_q: "queue.Queue[Line]" = queue.Queue()

    writer = TranscriptWriter(cfg)
    threads: list[threading.Thread] = []
    # Список, а не словарь по метке: выводов теперь сколько угодно, и все они
    # идут под одной меткой — свести их в словарь значит потерять все, кроме
    # последнего. Метка остаётся одна на всю дальнюю сторону сознательно:
    # разделение говорящих в задачи v1 не входит, и звук с двух гарнитур
    # относится к разговору ровно так же, как с одной.
    sources: list[tuple[str, str]] = []

    # Обязателен только первый вывод — на нём разговор. Микрофон, если он
    # вообще нашёлся, тоже обязателен: не открывшийся микрофон при живом
    # устройстве — это поломка, а не второй монитор с HDMI-звуком.
    channels: list[tuple[dict, str, bool]] = []
    if mic_device is None:
        sources.append((MIC_LABEL, t("app.no_device")))
    else:
        channels.append((mic_device, MIC_LABEL, True))
    for position, device in enumerate(loop_devices):
        channels.append((device, LOOPBACK_LABEL, position == 0))

    for device, label, required in channels:
        capture = AudioCapture(
            device, label, cfg, stop_event, pause_event, notifier, required=required
        )
        threads += [
            capture,
            VadSegmenter(capture, cfg, transcribe_q, stop_event, pause_event, notifier),
        ]
        sources.append((label, device["name"]))

    worker = TranscriberWorker(
        cfg, transcribe_q, gui_q, stop_event, notifier, writer, language, model
    )
    threads.append(worker)

    for thread in threads:
        thread.start()

    window = TranscriptWindow(
        cfg=cfg,
        gui_queue=gui_q,
        transcribe_queue=transcribe_q,
        notifier=notifier,
        stop_event=stop_event,
        pause_event=pause_event,
        worker=worker,
        sources=sources,
        language=language,
        model=model,
        store=store,
    )
    window.run()   # блокирует до закрытия окна пользователем

    # Даём распознаванию дожевать очередь: строки уже пишутся на диск
    # по мере готовности, даже когда окна больше нет.
    #
    # Бюджет один на всех, а не по 30 с на поток: своя пауза у каждого из
    # пяти означала бы до двух с половиной минут окна, которого уже нет на
    # экране. Ждать тут по-настоящему стоит только распознавание, поэтому
    # оно и стоит в очереди последним — остальные уходят за десятые доли
    # секунды и его времени не отнимают.
    deadline = time.monotonic() + 30
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    path = writer.finalize()
    if path:
        print(t("app.saved", path=path))
    else:
        print(t("app.nothing_saved"))
