"""Захват звука, который переживает смену устройств посреди звонка.

Раньше устройства выбирались один раз, при старте, и дальше приложение
слушало ровно их. За звонок это ломалось незаметно и всегда одинаково: у
одного из собеседников расшифровка прекращалась, у другого шла дальше, а
перезапуск «чинил». Перезапуск чинил потому, что заново открывал звук, —
и больше ничего. Причин у поломки несколько, все обычные:

  * Bluetooth-гарнитура переподключилась или переключила профиль
    (A2DP -> hands-free, как только звонилка открывает микрофон). Старая
    конечная точка исчезает, её поток больше не получает ни пакета;
  * звонилка перевела звук на другой выход, которого при старте не было;
  * Windows сменила устройство по умолчанию.

Поток захвата, чьё устройство пропало, либо останавливается, либо молчит.
Отличить «молчит, потому что собеседник молчит» от «молчит, потому что
устройства больше нет» по самому звуку нельзя: loopback тихого выхода не
отдаёт ни одного пакета. Поэтому смотрим на две вещи со стороны:

  * остановился ли поток, который до этого работал;
  * поменялся ли список устройств Windows (endpoints.endpoint_signature).

В обоих случаях захват пересоздаётся целиком — ровно то, что делал
перезапуск, только без перезапуска и без потери уже распознанного.
Целиком, а не по одному устройству: PortAudio перечитывает список
устройств только при полной переинициализации, то есть когда закрыт
последний поток. Пауза в захвате при этом — доли секунды, и приходится она
на момент, когда звук и так переключался.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from . import diagnostics
from .config import Config
from .i18n import speaker, t
from .models import Utterance
from .status import Notifier

_LOG = logging.getLogger("callcribe")

MIC_LABEL = "me"
LOOPBACK_LABEL = "them"

POLL_SECONDS = 1.0
"""Как часто сверяться со списком устройств. Сам опрос стоит пару
миллисекунд; реже — дольше дыра в расшифровке после переключения."""

SETTLE_SECONDS = 1.5
"""Сколько список должен простоять без изменений, прежде чем пересоздавать
захват. Bluetooth при переключении профиля роняет и поднимает устройства
несколькими шагами, и пересоздавать захват на каждом из них — значит
открыть то, что через полсекунды снова исчезнет."""

RETRY_SECONDS = (2.0, 5.0, 10.0, 30.0)
"""Паузы между неудачными попытками. Устройство, которое не открывается,
не должно превращать приложение в цикл из отказов раз в секунду."""

JOIN_SECONDS = 3.0

Sources = list[tuple[str, "str | None"]]
"""(метка канала, имя устройства); None — устройства для канала нет."""


@dataclass
class _Generation:
    """Один набор открытых устройств. Пересоздание — это новое поколение
    со своим событием остановки: так старое можно закрыть, не трогая
    общего «приложение закрывается»."""

    stop: threading.Event = field(default_factory=threading.Event)
    channels: list = field(default_factory=list)   # (capture, segmenter, required)
    sources: Sources = field(default_factory=list)

    def threads(self) -> list[threading.Thread]:
        out: list[threading.Thread] = []
        for capture, segmenter, _required in self.channels:
            out += [capture, segmenter]
        return out

    def broken(self) -> str | None:
        """Что сломалось, или None. Не открывшийся необязательный вывод —
        не поломка: он мог не работать и при старте (HDMI без монитора)."""
        if self.stop.is_set():
            return None
        for capture, segmenter, required in self.channels:
            if not capture.started.is_set():
                continue                    # ещё открывается
            if capture.opened and not capture.is_alive():
                return capture.device.get("name", "?")
            if required and not capture.opened:
                return capture.device.get("name", "?")
            if capture.opened and not segmenter.is_alive():
                return capture.device.get("name", "?")
        return None


def pick_devices(notifier: Notifier | None) -> tuple[dict | None, list[dict]]:
    """Микрофон по умолчанию и все выводы. Хотя бы один вывод обязателен,
    микрофон — нет: без него запись половины разговора всё равно
    полезнее, чем отказ стартовать.

    Вызывать из потока с COM-апартаментом. Под общим замком PortAudio:
    инициализация и завершение не потокобезопасны (см. audio.py)."""
    import pyaudiowpatch as pyaudio

    from .audio import _PORTAUDIO_LOCK, get_loopback_devices, get_mic_device

    with _PORTAUDIO_LOCK:
        pa = pyaudio.PyAudio()
        try:
            loopbacks = get_loopback_devices(pa)   # без них смысла нет — пусть падает
            try:
                mic = get_mic_device(pa)
            except Exception as exc:
                if notifier is not None:
                    notifier.warn(t("app.mic_unavailable", error=exc))
                mic = None
            return mic, loopbacks
        finally:
            pa.terminate()


class CaptureSupervisor(threading.Thread):
    """Держит захват открытым на тех устройствах, что есть СЕЙЧАС."""

    def __init__(
        self,
        cfg: Config,
        transcribe_queue: "queue.Queue[Utterance]",
        stop_event: threading.Event,
        pause_event: threading.Event,
        notifier: Notifier,
        *,
        enumerate_devices: Callable[[], tuple[dict | None, list[dict]]] | None = None,
        make_capture: Callable | None = None,
        make_segmenter: Callable | None = None,
        signature: Callable[[], object] | None = None,
        apartment: Callable | None = None,
    ):
        super().__init__(daemon=True, name="capture-supervisor")
        self.cfg = cfg
        self.transcribe_queue = transcribe_queue
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.notifier = notifier

        # Всё, что трогает настоящие устройства, подменяется: так логику
        # пересоздания можно проверить без звуковой карты (selftest).
        self._enumerate = enumerate_devices or (lambda: pick_devices(None))
        self._make_capture = make_capture or _default_capture
        self._make_segmenter = make_segmenter or _default_segmenter
        self._signature = signature or _default_signature
        self._apartment = apartment or _default_apartment

        self._generation: _Generation | None = None
        self._lock = threading.Lock()
        self.version = 0
        self.rebuilds = 0
        # Об отказе переподключиться говорим один раз на серию: повтор раз
        # в полминуты ничего нового человеку не сообщает.
        self._failure_reported = False

    # ------------------------------------------------------------------

    def sources(self) -> Sources:
        with self._lock:
            return list(self._generation.sources) if self._generation else []

    def _set_generation(self, generation: _Generation | None) -> None:
        with self._lock:
            self._generation = generation
            self.version += 1

    def _open(self, mic: dict | None, loopbacks: list[dict]) -> _Generation:
        """Открыть устройства. Обязателен первый вывод (на нём разговор) и
        микрофон, если он вообще нашёлся."""
        generation = _Generation()
        wanted: list[tuple[dict, str, bool]] = []
        if mic is None:
            generation.sources.append((MIC_LABEL, None))
        else:
            wanted.append((mic, MIC_LABEL, True))
        for position, device in enumerate(loopbacks):
            wanted.append((device, LOOPBACK_LABEL, position == 0))

        for device, label, required in wanted:
            capture = self._make_capture(
                device, label, self.cfg, generation.stop, self.pause_event,
                self.notifier, required,
            )
            segmenter = self._make_segmenter(
                capture, self.cfg, self.transcribe_queue, generation.stop,
                self.pause_event, self.notifier,
            )
            generation.channels.append((capture, segmenter, required))
            generation.sources.append((label, device.get("name", "?")))

        for thread in generation.threads():
            thread.start()
        return generation

    def _close(self, generation: _Generation | None) -> None:
        if generation is None:
            return
        generation.stop.set()
        deadline = time.monotonic() + JOIN_SECONDS
        # Сегментаторы — после захвата: им ещё дослать недосказанную фразу
        # в очередь распознавания, чтобы стык не съел слова.
        for capture, _segmenter, _required in generation.channels:
            capture.join(timeout=max(0.0, deadline - time.monotonic()))
        for _capture, segmenter, _required in generation.channels:
            segmenter.join(timeout=max(0.0, deadline - time.monotonic()))

    def begin(self, mic: dict | None, loopbacks: list[dict]) -> None:
        """Открыть первое поколение (устройства выбраны при старте, где
        отказ показывают окном) и начать следить."""
        self._set_generation(self._open(mic, loopbacks))
        self.start()

    # ------------------------------------------------------------------

    def _rebuild(self, reason: str) -> bool:
        before = self.sources()
        self._close(self._generation)
        try:
            mic, loopbacks = self._enumerate()
        except Exception as exc:
            diagnostics.log_exception("re-enumerating audio devices", exc)
            self._set_generation(None)
            if not self._failure_reported:
                self._failure_reported = True
                self.notifier.warn(t("audio.reconnect_failed", error=exc))
            return False

        generation = self._open(mic, loopbacks)
        self._set_generation(generation)
        self.rebuilds += 1
        self._failure_reported = False

        text = _describe(generation.sources)
        _LOG.info("audio rebuilt (%s): %s", reason, text)
        if generation.sources != before:
            self.notifier.warn(t("audio.reconnected", sources=text))
        else:
            self.notifier.info(t("audio.reconnected", sources=text))
        return True

    def run(self) -> None:
        try:
            with self._apartment():
                self._watch()
        except Exception as exc:
            # Наблюдатель умер — захват остался как есть, но следить за ним
            # больше некому. Это не повод ронять звонок, но сказать надо.
            diagnostics.log_exception("capture supervisor", exc)
            self.notifier.warn(t("audio.watch_failed", error=exc))
        finally:
            self._close(self._generation)

    def _watch(self) -> None:
        baseline = self._signature()
        latest, changed_at = baseline, None
        failures, retry_at = 0, 0.0
        # Когда пересоздание «удаётся», а канал снова тут же умирает (сбой,
        # который воспроизводится на первом же кадре), без этого вышло бы
        # пересоздание раз в секунду и поток предупреждений.
        recent: list[float] = []

        while not self.stop_event.wait(POLL_SECONDS):
            now = time.monotonic()
            current = self._signature()
            if current != latest:
                latest, changed_at = current, now
                _LOG.info("audio endpoints changed")

            reason = None
            generation = self._generation
            if generation is None:
                reason = "no devices"
            elif (
                changed_at is not None
                and now - changed_at >= SETTLE_SECONDS
                and latest is not None
                and latest != baseline
            ):
                reason = "devices changed"
            else:
                broken = generation.broken()
                if broken is not None:
                    reason = f"stream stopped: {broken}"

            if reason is None or now < retry_at:
                continue

            ok = self._rebuild(reason)
            baseline = self._signature()
            latest, changed_at = baseline, None
            now = time.monotonic()
            recent = [moment for moment in recent if now - moment < 60] + [now]
            if ok and len(recent) < 4:
                failures, retry_at = 0, 0.0
            else:
                pause = RETRY_SECONDS[min(failures, len(RETRY_SECONDS) - 1)]
                failures += 1
                retry_at = now + pause


def _describe(sources: Sources) -> str:
    grouped: dict[str, list[str]] = {}
    for label, name in sources:
        grouped.setdefault(label, []).append(name or t("app.no_device"))
    return " · ".join(f"{speaker(label)}: {', '.join(names)}" for label, names in grouped.items())


# --- настоящие устройства ---------------------------------------------------


def _default_capture(device, label, cfg, stop, pause, notifier, required):
    from .audio import AudioCapture

    return AudioCapture(device, label, cfg, stop, pause, notifier, required=required)


def _default_segmenter(capture, cfg, transcribe_queue, stop, pause, notifier):
    from .vad import VadSegmenter

    return VadSegmenter(capture, cfg, transcribe_queue, stop, pause, notifier)


def _default_signature():
    from .endpoints import endpoint_signature

    return endpoint_signature()


def _default_apartment():
    from .audio import _ComApartment

    return _ComApartment()
