"""Захват звука: микрофон и системный вывод через WASAPI loopback."""

from __future__ import annotations

import ctypes
import queue
import sys
import threading
import time

import numpy as np
import pyaudiowpatch as pyaudio

from .config import Config
from .i18n import speaker, t
from .resample import make_resampler
from .status import Notifier

# После скольких ошибок обработки в аудио-колбэке жаловаться пользователю.
_MAX_READ_ERRORS = 20

# Pa_Initialize() и Pa_Terminate() правят глобальное состояние PortAudio и
# потокобезопасными не являются. Пока источник был один, это не всплывало;
# стоит появиться микрофону — два потока захвата входят в инициализацию
# одновременно, и процесс падает с access violation ещё до появления окна.
# Под замком держим и открытие потока: оно тоже трогает структуры хост-API.
# Это только старт и остановка, на сам захват замок не влияет.
_PORTAUDIO_LOCK = threading.Lock()


class _ComApartment:
    """COM-апартамент для потока захвата.

    Pa_Initialize() считает ссылки, и настоящую инициализацию — вместе с
    CoInitialize — делает только ПЕРВЫЙ вызов. Второму потоку достаётся
    просто +1 к счётчику: апартамента у него нет, и открытие WASAPI-
    устройства падает с «Unanticipated host error» (-9999). С одним
    источником этого никто не видел, потому что поток был один.

    Поэтому апартамент заводим сами, до создания PyAudio. Модель — STA,
    та же, что берёт сам PortAudio, иначе получили бы RPC_E_CHANGED_MODE.
    """

    _S_OK = 0
    _S_FALSE = 1                  # апартамент уже был — но ссылку всё равно вернуть
    _APARTMENTTHREADED = 0x2

    def __init__(self) -> None:
        self._owned = False

    def __enter__(self) -> "_ComApartment":
        if sys.platform == "win32":
            hr = ctypes.windll.ole32.CoInitializeEx(None, self._APARTMENTTHREADED)
            # Всё прочее (RPC_E_CHANGED_MODE) — апартамент есть, но чужой
            # модели: работать он не мешает, а закрывать его не нам.
            self._owned = hr in (self._S_OK, self._S_FALSE)
        return self

    def __exit__(self, *exc: object) -> None:
        if self._owned:
            ctypes.windll.ole32.CoUninitialize()
            self._owned = False


# ---------------------------------------------------------------------------
# ВЫБОР УСТРОЙСТВ
# ---------------------------------------------------------------------------


def _wasapi(p: "pyaudio.PyAudio") -> dict:
    try:
        return p.get_host_api_info_by_type(pyaudio.paWASAPI)
    except OSError as exc:
        raise RuntimeError(t("audio.no_wasapi")) from exc


def get_mic_device(p: "pyaudio.PyAudio") -> dict:
    """Микрофон по умолчанию именно из WASAPI.

    get_default_input_device_info() возвращает устройство хост-API по
    умолчанию — на Windows это обычно MME, а не WASAPI. Смешивать хост-API
    между двумя источниками не стоит: у MME своя буферизация и своя
    задержка, и частота дискретизации может отличаться от реальной.
    """
    info = _wasapi(p)
    index = info.get("defaultInputDevice", -1)
    if index is not None and index >= 0:
        return p.get_device_info_by_index(index)

    try:
        fallback = p.get_default_input_device_info()
    except OSError:
        fallback = None
    if not fallback:
        raise RuntimeError(t("audio.no_mic"))
    return p.get_device_info_by_index(fallback["index"])


def get_loopback_device(p: "pyaudio.PyAudio") -> dict:
    """Loopback-устройство текущего вывода звука."""
    info = _wasapi(p)
    index = info.get("defaultOutputDevice", -1)
    if index is None or index < 0:
        raise RuntimeError(t("audio.no_output"))

    speakers = p.get_device_info_by_index(index)
    if speakers.get("isLoopbackDevice", False):
        return speakers

    for loopback in p.get_loopback_device_info_generator():
        if speakers["name"] in loopback["name"]:
            return loopback

    raise RuntimeError(t("audio.no_loopback", device=speakers["name"]))


def describe(device: dict) -> str:
    rate = int(device.get("defaultSampleRate", 0))
    channels = int(device.get("maxInputChannels", 0))
    return t("audio.describe", name=device["name"], rate=rate, channels=channels)


# ---------------------------------------------------------------------------
# ЗАХВАТ
# ---------------------------------------------------------------------------


class AudioCapture(threading.Thread):
    """Читает сырой звук с устройства и кладёт в очередь пары
    (время конца блока, float32 16 кГц mono).

    Метка времени ставится здесь, а не в VadSegmenter: только этот поток
    читает устройство в реальном времени. Если сегментатор на мгновение
    отстанет (всплеск нагрузки, старт приложения), время фраз всё равно
    останется верным.
    """

    def __init__(
        self,
        device: dict,
        label: str,
        cfg: Config,
        stop_event: threading.Event,
        pause_event: threading.Event,
        notifier: Notifier,
    ):
        super().__init__(daemon=True, name=f"capture-{label}")
        self.device = device
        self.label = label
        self.cfg = cfg
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.notifier = notifier

        self.native_rate = int(device["defaultSampleRate"])
        self.channels = max(1, int(device["maxInputChannels"]))
        self.read_frames = max(1, int(self.native_rate * cfg.read_ms / 1000))

        self.out_queue: "queue.Queue[tuple[float, np.ndarray]]" = queue.Queue()
        self.started = threading.Event()
        self._resampler = None
        self._callback_errors = 0

    # ------------------------------------------------------------------

    def _handle(self, raw: bytes, ts: float) -> None:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self.channels > 1:
            usable = len(samples) - len(samples) % self.channels
            samples = samples[:usable].reshape(-1, self.channels).mean(axis=1)

        resampled = self._resampler.process(samples)
        if len(resampled):
            self.out_queue.put((ts, resampled))

    def _callback(self, in_data, frame_count, time_info, status):
        if self.stop_event.is_set():
            return (None, pyaudio.paComplete)
        # На паузе просто выбрасываем данные: поток продолжает крутиться,
        # чтобы не переполнялся буфер драйвера.
        if in_data and not self.pause_event.is_set():
            try:
                self._handle(in_data, time.time())
            except Exception as exc:
                self._callback_errors += 1
                if self._callback_errors == _MAX_READ_ERRORS:
                    self.notifier.warn(
                        t("audio.process_errors", label=speaker(self.label), error=exc)
                    )
        return (None, pyaudio.paContinue)

    def run(self) -> None:
        # PyAudio создаётся ЗДЕСЬ, а не в __init__: WASAPI работает через COM,
        # а апартамент принадлежит конкретному потоку. Инициализировать
        # PortAudio в главном потоке и открывать поток в рабочем — источник
        # плавающих отказов на Windows.
        with _ComApartment():
            self._capture()

    def _capture(self) -> None:
        pa = None
        stream = None
        try:
            with _PORTAUDIO_LOCK:
                pa = pyaudio.PyAudio()
                self._resampler = make_resampler(self.native_rate, self.cfg.sample_rate_target)
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.native_rate,
                    input=True,
                    input_device_index=self.device["index"],
                    frames_per_buffer=self.read_frames,
                    stream_callback=self._callback,
                )
                stream.start_stream()
        except Exception as exc:
            self.notifier.fatal(
                t("audio.open_failed", label=speaker(self.label), error=exc)
            )
            # started выставить обязаны в любом случае: иначе тот, кто ждёт
            # старта источника, будет ждать его до таймаута впустую.
            try:
                with _PORTAUDIO_LOCK:
                    if stream is not None:
                        stream.close()
                    if pa is not None:
                        pa.terminate()
            except Exception:
                pass
            self.started.set()
            return

        self.started.set()
        try:
            # Ждём именно остановки, а не данных. Блокирующий stream.read()
            # здесь не годится: loopback молчащего устройства вывода не отдаёт
            # НИ ОДНОГО пакета, и чтение зависает до следующего звука. Окно
            # закрывают как раз в тишине — приложение висло бы каждый раз.
            while not self.stop_event.wait(0.1):
                if not stream.is_active():
                    self.notifier.warn(t("audio.stopped", label=speaker(self.label)))
                    break
        finally:
            with _PORTAUDIO_LOCK:
                try:
                    stream.stop_stream()
                    stream.close()
                except Exception:
                    pass
                # Освобождаем ресемплер до финализации интерпретатора, иначе
                # nanobind внутри soxr печатает "leaked instance" при выходе.
                self._resampler = None
                pa.terminate()
