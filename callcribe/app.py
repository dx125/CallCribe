"""Сборка приложения и главный цикл."""

from __future__ import annotations

import dataclasses
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox

from . import diagnostics
from .asr import TranscriberWorker
from .audio import describe
from .capture import LOOPBACK_LABEL, MIC_LABEL, CaptureSupervisor, pick_devices
from .config import CFG, Config, DeviceSetting, LanguageSetting, ModelSetting, language_name
from .i18n import set_language, t, ui_language_name
from .models import Line, Utterance
from .resample import HAVE_SOXR
from .settings import SettingsStore, model_display
from .status import Notifier
from .transcript import TranscriptWriter
from .ui import TranscriptWindow

# Метки каналов — ключи, а не готовые подписи: они попадают и в окно, и в
# файл, а язык интерфейса меняется на ходу. Перевод берётся в момент
# показа, см. i18n.speaker(). Определены в capture.py, здесь — для тех,
# кто привык брать их отсюда.
__all__ = ["MIC_LABEL", "LOOPBACK_LABEL", "load_settings", "run"]


def _show_startup_error(message: str) -> None:
    root = tk.Tk()
    root.withdraw()
    messagebox.showerror(t("app.audio_error_title"), message)
    root.destroy()


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
        whisper_device=store.data.whisper_device,
    )
    return store, cfg, problems


def run(
    cfg: Config | None = None,
    store: SettingsStore | None = None,
    problems: list[tuple[str, dict]] | None = None,
) -> None:
    cfg = cfg or CFG
    notifier = Notifier()
    # Поток, умерший от исключения, должен дойти до окна, а не только до файла.
    diagnostics.set_fatal_sink(notifier.fatal)

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

    diagnostics.set_phase("audio")
    try:
        mic_device, loop_devices = pick_devices(notifier)
    except Exception as exc:
        diagnostics.log_exception("picking audio devices", exc)
        _show_startup_error(str(exc))
        sys.exit(diagnostics.EXIT_HANDLED)

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
    device = DeviceSetting(cfg.whisper_device)

    stop_event = threading.Event()
    pause_event = threading.Event()
    transcribe_q: "queue.Queue[Utterance]" = queue.Queue()
    gui_q: "queue.Queue[Line]" = queue.Queue()

    writer = TranscriptWriter(cfg)

    # Захват живёт отдельным хозяйством: устройства за звонок меняются
    # (Bluetooth переподключился, звонилка сменила выход), и он умеет
    # открыть их заново сам — см. capture.py.
    capture = CaptureSupervisor(cfg, transcribe_q, stop_event, pause_event, notifier)
    capture.begin(mic_device, loop_devices)

    worker = TranscriberWorker(
        cfg, transcribe_q, gui_q, stop_event, notifier, writer, language, model, device
    )
    worker.producer = capture
    worker.start()

    window = TranscriptWindow(
        cfg=cfg,
        gui_queue=gui_q,
        transcribe_queue=transcribe_q,
        notifier=notifier,
        stop_event=stop_event,
        pause_event=pause_event,
        worker=worker,
        sources=capture.sources(),
        language=language,
        model=model,
        store=store,
        device=device,
        capture=capture,
    )
    window.run()   # блокирует до закрытия окна пользователем
    diagnostics.set_phase("shutdown")

    # Даём распознаванию дожевать очередь: строки уже пишутся на диск
    # по мере готовности, даже когда окна больше нет.
    #
    # Бюджет один на всех: ждать тут по-настоящему стоит только
    # распознавание, поэтому оно и последнее — захват закрывается за
    # доли секунды и дописывает в очередь недосказанные фразы.
    deadline = time.monotonic() + 30
    for thread in (capture, worker):
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    path = writer.finalize()
    if path:
        print(t("app.saved", path=path))
    else:
        print(t("app.nothing_saved"))
