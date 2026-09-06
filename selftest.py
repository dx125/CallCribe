#!/usr/bin/env python3
"""Проверка машины и конвейера без реального звонка.

    python selftest.py

Гоняет синтетический звук через ресемплер и VAD, проверяет фильтр
галлюцинаций, показывает найденные устройства и то, куда встанет whisper
(GPU или CPU). Модель не грузит — это быстро.

    python selftest.py --load-model     # ещё и загрузить whisper и
                                        # распознать тестовый сигнал
"""

from __future__ import annotations

import argparse
import contextlib
import io
import queue
import sys
import threading
import time
from collections import Counter

import numpy as np

from callcribe import filters
from callcribe.config import Config
from callcribe.models import Utterance
from callcribe.resample import HAVE_SOXR, make_resampler
from callcribe.status import use_utf8_console

PASSED = 0
FAILED = 0
WARNED = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASSED, FAILED
    mark = "OK  " if condition else "FAIL"
    if condition:
        PASSED += 1
    else:
        FAILED += 1
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def warn(name: str, detail: str = "") -> None:
    """Не отказ — приложение это переживёт, но пользователю стоит знать."""
    global WARNED
    WARNED += 1
    print(f"  [ВНИМ] {name}" + (f" — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------------------


def voiced_signal(seconds: float, sr: int, f0: float = 120.0) -> np.ndarray:
    """Гармонический сигнал, похожий на озвонченную речь — webrtcvad
    распознаёт его как речь, в отличие от чистой синусоиды."""
    t = np.arange(int(seconds * sr)) / sr
    signal = np.zeros_like(t)
    for harmonic in range(1, 25):
        signal += np.sin(2 * np.pi * f0 * harmonic * t) / harmonic
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


def chirp(seconds: float, sr: int, f0: float = 100.0, f1: float = 170.0) -> np.ndarray:
    """Речеподобный сигнал, в котором нет двух одинаковых 30-мс кадров.

    Именно это отличает его от voiced_signal: там сигнал периодический, все
    кадры побитово совпадают, и «сегментатор отдал один и тот же звук
    дважды» в нём принципиально не отличить от «сигнал сам такой». На
    скользящей частоте каждый кадр уникален, и повтор виден точно.
    """
    t = np.arange(int(seconds * sr)) / sr
    instant = f0 + (f1 - f0) * t / max(t[-1], 1e-9)
    phase = 2 * np.pi * np.cumsum(instant) / sr
    signal = np.zeros_like(t)
    for harmonic in range(1, 25):
        signal += np.sin(harmonic * phase) / harmonic
    return (0.3 * signal / np.max(np.abs(signal))).astype(np.float32)


def tone(seconds: float, sr: int, freq: float) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) if len(x) else 0.0


def dominant_freq(x: np.ndarray, sr: int) -> float:
    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.fft.rfftfreq(len(x), 1 / sr)[int(np.argmax(spectrum))])


def feed_chunks(resampler, signal: np.ndarray, chunk: int) -> np.ndarray:
    out = [resampler.process(signal[i : i + chunk]) for i in range(0, len(signal), chunk)]
    return np.concatenate(out) if out else np.zeros(0, np.float32)


# ---------------------------------------------------------------------------


def test_resampler() -> None:
    section("Ресемплинг 48000 -> 16000")
    print(f"  движок: {'libsoxr' if HAVE_SOXR else 'numpy-фолбэк'}")
    sr_in, sr_out, chunk = 48_000, 16_000, 4_800

    signal = tone(1.0, sr_in, 1_000.0)
    out = feed_chunks(make_resampler(sr_in, sr_out), signal, chunk)
    # Потоковый ресемплер держит внутри задержку фильтра (у soxr HQ это
    # ~340 сэмплов / 21 мс на 16 кГц) — она выйдет со следующим блоком.
    # Недостача на конец потока ожидаема, лишних сэмплов быть не должно.
    lag = sr_out - len(out)
    check("длина выхода", 0 <= lag < 600, f"{len(out)} сэмплов, задержка фильтра {lag}")
    check(
        "тон 1 кГц сохранён",
        abs(dominant_freq(out, sr_out) - 1_000.0) < 30,
        f"пик на {dominant_freq(out, sr_out):.0f} Гц",
    )
    check("уровень не потерян", 0.5 < rms(out) / rms(signal) < 1.5, f"RMS x{rms(out)/rms(signal):.2f}")

    # 12 кГц выше Найквиста для 16 кГц. Без ФНЧ он завернётся в 4 кГц
    # с полной энергией — именно этого мы и не хотим.
    alias_in = tone(1.0, sr_in, 12_000.0)
    alias_out = feed_chunks(make_resampler(sr_in, sr_out), alias_in, chunk)
    ratio = rms(alias_out) / rms(alias_in)
    check("12 кГц подавлен, а не завёрнут в 4 кГц", ratio < 0.1, f"осталось {ratio*100:.1f}% энергии")

    # Стык чанков не должен давать щелчков: непрерывный тон, поданный
    # блоками, обязан остаться непрерывным.
    smooth = feed_chunks(make_resampler(sr_in, sr_out), tone(1.0, sr_in, 440.0), chunk)
    jumps = np.max(np.abs(np.diff(smooth[100:-100]))) if len(smooth) > 400 else 0
    expected_step = 2 * np.pi * 440 / sr_out * 0.5
    check("нет разрывов на стыках блоков", jumps < expected_step * 3, f"макс. скачок {jumps:.4f}")


def test_filters() -> None:
    section("Фильтр галлюцинаций")
    garbage = [
        "Продолжение следует...",
        "Субтитры сделал DimaTorzok",
        "Спасибо за просмотр!",
        "Подписывайтесь на канал!",
        "Thanks for watching!",
        "Subtitles by the amara.org community",
        "да да да да да да да да",
        "так так так так так так так так так так",
        ".",
        "  ",
        # Испанские штампы того же корпуса субтитров: без них режим
        # "Español" сыпал бы в расшифровку ровно этот мусор.
        "¡Gracias por ver el video!",
        "Suscríbete al canal",
        "Subtítulos realizados por la comunidad de Amara.org",
        "Hasta la próxima",
    ]
    for text in garbage:
        check(f"отсеяно: {text!r}", filters.is_hallucination(text))

    real = [
        "Давай вынесем это в отдельный middleware",
        "У нас там EF Core миграция не накатилась",
        "Ok, let's merge it after the code review",
        "Ага",
        "Нет, refresh token живёт тридцать дней",
        "Vamos a mover la autorización a un middleware aparte",
        "El refresh token dura treinta días",
        # Границы испанских маркеров: они длинные не случайно — разговор
        # про локализацию не должен пропадать целиком.
        "Hay que traducir los subtítulos de la documentación",
    ]
    for text in real:
        check(f"пропущено: {text!r}", not filters.is_hallucination(text))


def test_languages() -> None:
    """Выбор языка: список, промпты и настройка, которую правят на ходу."""
    section("Языки")
    from callcribe.config import (
        AUTO,
        KNOWN_LANGUAGES,
        LANGUAGE_CODES,
        MAX_PROMPT_CHARS,
        LanguageSetting,
        language_code,
        language_name,
        language_names,
    )

    cfg = Config()
    check(
        "в списке четыре языка",
        list(LANGUAGE_CODES) == ["ru", "en", "es", AUTO],
        ", ".join(language_names()),
    )
    check("«Авто» не считается известным языком", AUTO not in KNOWN_LANGUAGES,
          f"известные: {', '.join(sorted(KNOWN_LANGUAGES))}")

    # Подпись <-> код в обе стороны: на этом держится и выпадающий список,
    # и разбор --lang.
    broken = [language_name(code) for code in LANGUAGE_CODES
              if language_code(language_name(code)) != code]
    check("подпись и код переводятся друг в друга", not broken,
          ", ".join(broken) or "все четыре")

    # Названия языков — на них самих, и от языка интерфейса не зависят:
    # человек ищет в списке знакомое начертание. Переводится только «Авто».
    from callcribe import i18n

    was = i18n.get_language()
    try:
        i18n.set_language("en")
        english = language_names()
        i18n.set_language("ru")
        russian = language_names()
    finally:
        i18n.set_language(was)
    check(
        "названия языков не переводятся, кроме «Авто»",
        english[:3] == russian[:3] == ["Русский", "English", "Español"]
        and english[3] != russian[3],
        f"{english[3]!r} / {russian[3]!r}",
    )

    # Промпт нужен каждому языку, включая «Авто», иначе термины перестают
    # подсказываться молча.
    missing = [language_name(code) for code in LANGUAGE_CODES if not cfg.prompt_for(code)]
    check("у каждого языка есть промпт", not missing, ", ".join(missing) or "все четыре")

    too_long = [
        f"{language_name(code)}: {len(cfg.prompt_for(code))}"
        for code in LANGUAGE_CODES
        if len(cfg.prompt_for(code)) > MAX_PROMPT_CHARS
    ]
    check(
        f"промпты укладываются в {MAX_PROMPT_CHARS} символов",
        not too_long,
        ", ".join(too_long)
        or ", ".join(f"{language_name(c)}: {len(cfg.prompt_for(c))}" for c in LANGUAGE_CODES),
    )

    # Промпты обязаны различаться: русский список слов при испанской речи
    # тянет вывод обратно в русский — ради этого всё и разделялось.
    check(
        "промпт у каждого языка свой",
        len({cfg.prompt_for(code) for code in LANGUAGE_CODES}) == len(LANGUAGE_CODES),
        f"различных промптов: {len({cfg.prompt_for(c) for c in LANGUAGE_CODES})}",
    )
    cyrillic = any("а" <= ch.lower() <= "я" for ch in cfg.prompt_for(AUTO))
    check("в промпте «Авто» нет слов конкретного языка", not cyrillic,
          "только имена технологий")
    check("неизвестный код откатывается на промпт «Авто»",
          cfg.prompt_for("de") == cfg.prompt_for(AUTO))

    # Настройка, которую окно правит во время звонка.
    setting = LanguageSetting("ru")
    setting.set(AUTO)
    check("язык переключается на ходу", setting.get() is AUTO, "ru -> Авто")

    # validate() должен ловить язык, которого нет в окне: выбрать его
    # обратно после переключения будет уже нельзя.
    complaints = [w for w in Config(language="de").validate() if "language" in w]
    check("validate() ругается на язык вне списка", bool(complaints),
          complaints[0][:60] if complaints else "промолчал")
    check("на каждый язык из списка validate() молчит",
          not any(Config(language=code).validate() for code in LANGUAGE_CODES),
          "ru, en, es, Авто")


class _FakeSegments:
    """Пустой результат распознавания, умеющий сказать, закрыли ли его."""

    def __init__(self) -> None:
        self.closed = False

    def __iter__(self):
        return iter(())

    def close(self) -> None:
        self.closed = True


class _FakeInfo:
    def __init__(self, language: str | None, all_probs) -> None:
        self.language = language
        self.language_probability = 0.87
        self.all_language_probs = list(all_probs)


class _FakeModel:
    """Заглушка whisper: запоминает, с чем её позвали, и ничего не считает.

    Нужна, чтобы проверить маршрутизацию языка без загрузки модели. Трогать
    её из главного потока безопасно ровно потому, что настоящей модели
    CTranslate2 здесь нет: у той обращение извне её потока роняет процесс.
    """

    def __init__(self, detected: str = "ru", all_probs=()) -> None:
        self.calls: list[dict] = []
        self.segments: list[_FakeSegments] = []
        self._detected = detected
        self._all_probs = all_probs

    def transcribe(self, audio, **kwargs):
        self.calls.append(kwargs)
        language = kwargs["language"]
        segments = _FakeSegments()
        self.segments.append(segments)
        # Язык, заданный явно, faster-whisper и возвращает как есть;
        # определяет он его только при language=None.
        detected = self._detected if language is None else language
        return segments, _FakeInfo(detected, self._all_probs)


def _route(language_setting, detected="ru", all_probs=(), **cfg_kw):
    """Прогнать одну фразу через worker с заглушкой вместо модели."""
    from callcribe.asr import TranscriberWorker
    from callcribe.status import Notifier
    from callcribe.transcript import TranscriptWriter

    cfg = Config(**cfg_kw)
    notifier = Notifier()
    worker = TranscriberWorker(
        cfg, queue.Queue(), queue.Queue(), threading.Event(), notifier,
        TranscriptWriter(cfg), language_setting,
    )
    worker.model = _FakeModel(detected, all_probs)
    # Жалобы заглушки печатать некуда: в selftest это выглядело бы отказом,
    # хотя проверяется как раз штатная правка промаха детектора.
    with contextlib.redirect_stderr(io.StringIO()):
        worker._transcribe(Utterance("Тест", time.time(), np.zeros(8_000, np.float32)))
    return worker.model, notifier


def test_language_routing() -> None:
    """Доходит ли выбранный язык до модели — и с каким промптом.

    Проверяется без whisper: важна не точность распознавания, а то, что
    выбор из окна превращается в правильные аргументы вызова.
    """
    section("Язык доходит до модели")
    from callcribe.config import AUTO, LanguageSetting
    from callcribe.status import INFO

    cfg = Config()

    # Явный язык — ровно один вызов, промпт этого языка.
    for code in ("ru", "en", "es"):
        model, _ = _route(LanguageSetting(code))
        ok = (
            len(model.calls) == 1
            and model.calls[0]["language"] == code
            and model.calls[0]["initial_prompt"] == cfg.prompt_for(code)
        )
        check(f"язык {code!r} уходит в модель со своим промптом", ok,
              f"вызовов: {len(model.calls)}, язык: {model.calls[0]['language']!r}")

    # «Авто»: сначала определение (language=None), потом расшифровка уже
    # с конкретным языком — ради промпта этого языка.
    model, _ = _route(LanguageSetting(AUTO), detected="en")
    check(
        "«Авто» определяет язык, а расшифровывает уже явным",
        [call["language"] for call in model.calls] == [None, "en"],
        " -> ".join(repr(call["language"]) for call in model.calls),
    )
    check(
        "после определения берётся промпт найденного языка",
        model.calls[-1]["initial_prompt"] == cfg.prompt_for("en"),
    )
    check(
        "пробный проход закрыт, декодирования на нём не было",
        model.segments[0].closed and not model.segments[1].closed,
        "первый генератор закрыт, второй прочитан",
    )

    # Промах детектора: валлийский на секундной русской реплике — ровно то,
    # что whisper делает регулярно. Должны подставить ближайший из списка.
    probs = [("cy", 0.41), ("ru", 0.29), ("en", 0.11), ("es", 0.04)]
    model, notifier = _route(LanguageSetting(AUTO), detected="cy", all_probs=probs)
    check(
        "язык вне списка заменён ближайшим из списка",
        [call["language"] for call in model.calls] == [None, "ru"],
        " -> ".join(repr(call["language"]) for call in model.calls),
    )
    complaints = []
    while True:
        try:
            notice = notifier.queue.get_nowait()
        except queue.Empty:
            break
        if notice.level != INFO:
            complaints.append(notice.text)
    check("о подмене языка пользователю сказано", bool(complaints),
          complaints[0][:70] if complaints else "промолчали")

    # Тот же промах при выключенной правке — доверяем детектору как есть.
    model, _ = _route(
        LanguageSetting(AUTO), detected="cy", all_probs=probs, restrict_auto_language=False
    )
    check(
        "с restrict_auto_language=False детектор не правится",
        [call["language"] for call in model.calls] == [None, "cy"],
        " -> ".join(repr(call["language"]) for call in model.calls),
    )

    # Переключение во время звонка: worker читает настройку на каждой фразе.
    setting = LanguageSetting("ru")
    model, _ = _route(setting)
    setting.set("es")
    model2, _ = _route(setting)
    check(
        "переключение в окне подхватывается следующей фразой",
        model.calls[0]["language"] == "ru" and model2.calls[0]["language"] == "es",
        f"{model.calls[0]['language']!r} -> {model2.calls[0]['language']!r}",
    )


class _BuildLog:
    """Заглушка загрузки модели: помнит, в каком состоянии её позвали."""

    def __init__(self, worker, failing=()):
        self.worker = worker
        self.failing = set(failing)
        self.calls: list[tuple[str, str | None]] = []
        self.model_seen: list[object] = []
        self.loading_seen: list[bool] = []

    def __call__(self, name: str, cpu_fallback: str | None):
        self.calls.append((name, cpu_fallback))
        self.model_seen.append(self.worker.model)
        self.loading_seen.append(self.worker.loading.is_set())
        if name in self.failing:
            raise RuntimeError(f"папки нет: {name}")
        return _FakeModel(), name, f"{name} · cpu/int8"


def _switch_worker(model_name: str, failing=()):
    """Поток распознавания с заглушкой вместо настоящей загрузки модели."""
    from callcribe.asr import TranscriberWorker
    from callcribe.config import ModelSetting
    from callcribe.status import Notifier
    from callcribe.transcript import TranscriptWriter

    cfg = Config(whisper_model=model_name)
    setting = ModelSetting(model_name)
    notifier = Notifier()
    worker = TranscriberWorker(
        cfg, queue.Queue(), queue.Queue(), threading.Event(), notifier,
        TranscriptWriter(cfg), None, setting,
    )
    worker.model = _FakeModel()
    worker.device_label = f"{model_name} · cpu/int8"
    worker._build_model = _BuildLog(worker, failing)
    return worker, setting, notifier


def _drain(notifier) -> list[tuple[str, str]]:
    from callcribe.status import INFO

    out = []
    while True:
        try:
            notice = notifier.queue.get_nowait()
        except queue.Empty:
            return out
        if notice.level != INFO:
            out.append((notice.level, notice.text))


def test_model_switch() -> None:
    """Смена модели на ходу: заявка из окна, работа — в потоке модели."""
    section("Смена модели")
    from callcribe.config import ModelSetting

    setting = ModelSetting("large-v3")
    setting.request("large-v3")
    check("выбор уже выбранного не заводит заявку", setting.pending() is None)
    setting.request("small")
    check("заявка видна до того, как её забрали", setting.pending() == "small")
    check("забрать заявку можно один раз",
          (setting.take(), setting.take()) == ("small", None))
    check("до подтверждения стоит прежняя модель", setting.get() == "large-v3",
          "загрузка ещё идёт")
    setting.confirm("small")
    check("после подтверждения — новая", setting.get() == "small")

    # --- удачное переключение ------------------------------------------
    worker, setting, notifier = _switch_worker("large-v3")
    worker._apply_pending_model()
    check("без заявки поток модель не трогает", not worker._build_model.calls,
          f"вызовов загрузки: {len(worker._build_model.calls)}")

    setting.request("small")
    with contextlib.redirect_stderr(io.StringIO()):
        worker._apply_pending_model()
    log = worker._build_model
    check("заявка исполнена", setting.get() == "small" and log.calls[0][0] == "small",
          f"стоит {setting.get()}")
    check("подпись устройства обновлена", "small" in worker.device_label,
          worker.device_label)
    check("«гружусь» снято по завершении", not worker.loading.is_set())
    check("модель на месте", worker.model is not None)

    # Две large-v3 разом — это лишние 3 ГБ, и на видеокарте средних
    # размеров переключение просто не влезет. Старую отпускаем ДО загрузки.
    check("прежняя модель отпущена до загрузки новой", log.model_seen[0] is None,
          "в момент загрузки модели в работе нет")
    # Окно опрашивает состояние каждые 100 мс: если между «заявка забрана»
    # и «пошла загрузка» есть щель, оно успевает решить, что всё провалилось.
    check("«гружусь» взведено на всё время загрузки", log.loading_seen[0] is True)
    # Явный выбор пользователя подменять нельзя: откат на turbo при старте
    # спасает от неподъёмной модели, а здесь он был бы враньём в подписи.
    check("явный выбор не подменяется откатом на CPU", log.calls[0][1] is None,
          f"cpu_fallback={log.calls[0][1]!r}")

    # --- подпись и откат на CPU: настоящий _build_model ------------------
    # Здесь заглушкой подменяется не метод, а сама faster_whisper: только
    # так проверяется то, что метод действительно делает, а не то, что
    # заглушка вернула.
    import dataclasses
    import types

    fake = types.ModuleType("faster_whisper")
    saved = sys.modules.get("faster_whisper")
    sys.modules["faster_whisper"] = fake
    try:
        fake.WhisperModel = lambda name, device=None, compute_type=None: object()
        worker, _setting, _notifier = _switch_worker("large-v3")
        del worker._build_model          # вернуть настоящий метод класса
        worker.cfg = dataclasses.replace(worker.cfg, whisper_device="cpu")

        path = r"D:\models\faster-whisper-large-v3"
        with contextlib.redirect_stderr(io.StringIO()):
            _model, loaded, label = worker._build_model(path, None)
        # Путь целиком в подпись не влезает: в окне она стоит в одной
        # строке со статусом и вытеснила бы всё остальное.
        check("в подписи устройства от пути остаётся имя папки",
              label == "faster-whisper-large-v3 · cpu/int8", label)
        check("грузится при этом полный путь, а не подпись", loaded == path, loaded)

        def only_cpu(name, device=None, compute_type=None):
            if device == "cuda":
                raise RuntimeError("cuDNN не найден")
            return object()

        fake.WhisperModel = only_cpu
        worker.cfg = dataclasses.replace(worker.cfg, whisper_device="cuda")
        with contextlib.redirect_stderr(io.StringIO()):
            _model, loaded, _label = worker._build_model("large-v3", "large-v3-turbo")
        # При старте откат осмыслен: на CPU large-v3 не успевает за речью.
        check("откат на CPU при старте берёт cpu_fallback_model",
              loaded == "large-v3-turbo", loaded)
        with contextlib.redirect_stderr(io.StringIO()):
            _model, loaded, _label = worker._build_model("large-v3", None)
        # А при смене из окна подменять выбор нельзя: человек выбрал явно.
        check("откат при смене модели оставляет выбранную",
              loaded == "large-v3", loaded)

        # Машина без видеокарты. Ошибки загрузки тут нет вовсе — large-v3
        # на CPU поднимется и будет молча отставать, поэтому подмена
        # должна случиться заранее, а не по исключению.
        fake.WhisperModel = lambda name, device=None, compute_type=None: object()
        worker.cfg = dataclasses.replace(worker.cfg, whisper_device="cpu")
        with contextlib.redirect_stderr(io.StringIO()):
            _model, loaded, _label = worker._build_model("large-v3", "large-v3-turbo")
        check("старт без видеокарты берёт cpu_fallback_model сразу",
              loaded == "large-v3-turbo", loaded)

        # ...но выбранную вручную папку подменять нечем: имени размера у
        # неё нет, а cpu_fallback_model полез бы качать другую модель.
        with contextlib.redirect_stderr(io.StringIO()):
            _model, loaded, _label = worker._build_model(path, "large-v3-turbo")
        check("свою папку на CPU не подменяет", loaded == path, loaded)
    finally:
        if saved is not None:
            sys.modules["faster_whisper"] = saved
        else:
            sys.modules.pop("faster_whisper", None)

    # --- видеокарта видна, а считать нечем -------------------------------
    # Драйвер ставится отдельно от cuBLAS/cuDNN, поэтому «устройство есть»
    # и «на нём можно считать» — разные факты. Раньше приложение верило
    # первому и роняло каждую фразу.
    import callcribe.asr as asr_module

    worker, _setting, notifier = _switch_worker("large-v3")
    saved_count = asr_module.cuda_device_count
    saved_missing = asr_module.missing_cuda_libraries
    try:
        asr_module.cuda_device_count = lambda: 1

        asr_module.missing_cuda_libraries = lambda: ["cublas64_12.dll"]
        worker.cfg = dataclasses.replace(worker.cfg, whisper_device="auto")
        with contextlib.redirect_stderr(io.StringIO()):
            device, _compute = worker._resolve_device()
        check("без библиотек счёта «auto» уходит на CPU", device == "cpu", device)
        said = [text for _level, text in _drain(notifier)]
        check("и говорит, чего не хватает и как починить",
              any("cublas64_12.dll" in m and "requirements-gpu" in m for m in said),
              said[-1] if said else "ни слова")

        # Явный выбор не подменяем — но предупреждение уже прозвучало.
        worker.cfg = dataclasses.replace(worker.cfg, whisper_device="cuda")
        with contextlib.redirect_stderr(io.StringIO()):
            device, _compute = worker._resolve_device()
        check("выбранное руками cuda остаётся cuda", device == "cuda", device)

        asr_module.missing_cuda_libraries = lambda: []
        worker.cfg = dataclasses.replace(worker.cfg, whisper_device="auto")
        with contextlib.redirect_stderr(io.StringIO()):
            device, compute = worker._resolve_device()
        check("с библиотеками на месте «auto» берёт видеокарту",
              (device, compute) == ("cuda", "float16"), f"{device}/{compute}")
    finally:
        asr_module.cuda_device_count = saved_count
        asr_module.missing_cuda_libraries = saved_missing

    # --- модель не поднялась -------------------------------------------
    worker, setting, notifier = _switch_worker("large-v3", failing={"мусор"})
    setting.request("мусор")
    with contextlib.redirect_stderr(io.StringIO()):
        worker._apply_pending_model()
    problems = _drain(notifier)
    check("на неудачной модели остаёмся на прежней", setting.get() == "large-v3",
          f"стоит {setting.get()}")
    check("прежняя модель поднята обратно",
          worker.model is not None and "large-v3" in worker.device_label,
          worker.device_label)
    check("распознавание при этом не остановлено", not worker.failed.is_set())
    check("о неудаче сказано предупреждением",
          any(level == "warn" for level, _ in problems),
          problems[0][1][:60] if problems else "промолчали")
    check("«гружусь» снято и после неудачи", not worker.loading.is_set())

    # --- не поднялась и прежняя ------------------------------------------
    worker, setting, notifier = _switch_worker("large-v3", failing={"мусор", "large-v3"})
    setting.request("мусор")
    with contextlib.redirect_stderr(io.StringIO()):
        worker._apply_pending_model()
    problems = _drain(notifier)
    check("потеря обеих моделей — это fatal, а не тишина",
          any(level == "fatal" for level, _ in problems),
          problems[-1][1][:70] if problems else "промолчали")
    check("поток помечен отказавшим", worker.failed.is_set(),
          "фразы дальше не берутся")
    check("«гружусь» снято и в этом случае", not worker.loading.is_set())


def _string_constants(path) -> set[str]:
    """Все строковые литералы модуля — чтобы найти ключи каталога в коде."""
    import ast

    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
    return found


def _translated_keys(path) -> set[str]:
    """Ключи, с которыми в модуле реально зовут t(...)."""
    import ast

    keys: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(
            node.func, "attr", ""
        )
        if name != "t":
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            keys.add(first.value)
    return keys


def test_i18n() -> None:
    """Каталог надписей: полон, согласован и сходится с кодом."""
    section("Язык интерфейса")
    import pathlib

    from callcribe import i18n
    from callcribe.i18n import (
        MESSAGES,
        UI_LANGUAGE_CODES,
        Translator,
        placeholders,
        speaker,
        ui_language_code,
        ui_language_name,
    )

    check("язык интерфейса по умолчанию — английский",
          i18n.DEFAULT_UI_LANGUAGE == "en", f"всего языков: {len(UI_LANGUAGE_CODES)}")

    # Ключ без одного из языков — это надпись, которая молча выйдет
    # по-английски посреди русского окна.
    incomplete = [
        key for key, entry in MESSAGES.items()
        if any(not entry.get(code) for code in UI_LANGUAGE_CODES)
    ]
    check(f"у всех {len(MESSAGES)} ключей есть оба языка", not incomplete,
          ", ".join(incomplete[:3]) or "ru и en везде")

    # Разные подстановки в переводах одного ключа — это отсутствующее
    # значение в сообщении: где-то {error} есть, а где-то потерялся.
    mismatched = [
        key for key, entry in MESSAGES.items()
        if len({placeholders(entry[code]) for code in UI_LANGUAGE_CODES if entry.get(code)}) > 1
    ]
    check("подстановки в переводах совпадают", not mismatched,
          ", ".join(mismatched[:3]) or f"сверено ключей: {len(MESSAGES)}")

    # Код и каталог должны сходиться в обе стороны. Опечатка в ключе
    # иначе видна только тому, кто наткнётся на это сообщение вживую.
    package = sorted(pathlib.Path("callcribe").glob("*.py"))
    used: set[str] = set()
    literals: set[str] = set()
    for module in package:
        used |= _translated_keys(module)
        literals |= _string_constants(module)

    unknown = sorted(used - set(MESSAGES))
    check("все ключи из кода есть в каталоге", not unknown,
          ", ".join(unknown[:3]) or f"вызовов t(): {len(used)}")

    dead = sorted(set(MESSAGES) - literals)
    check("в каталоге нет забытых ключей", not dead,
          ", ".join(dead[:3]) or "все используются")

    # Переводчик обязан пережить и опечатку в ключе, и нехватку значения:
    # он составляет в том числе сообщения об ошибках, и падать в этот
    # момент — худшее, что он может сделать.
    # Сверяемся с ожидаемым текстом, а не с самим каталогом: сравнение
    # каталога с собой прошло бы и на пустом каталоге.
    tr = Translator("en")
    check("неизвестный ключ возвращается как есть", tr("нет.такого") == "нет.такого")
    check("нехватка подстановки не роняет перевод",
          "{device}" in tr("asr.loading", model="x"), "вернулся шаблон целиком")
    tr.set("de")
    check("неизвестный язык откатывается на английский", tr.get() == "en")

    tr = Translator("ru")
    check("перевод берётся по выбранному языку", tr("ui.pause") == "Пауза", tr("ui.pause"))
    tr.set("en")
    check("язык переключается на ходу", tr("ui.pause") == "Pause", tr("ui.pause"))

    # Подпись <-> код языка интерфейса — на этом держится выпадающий список.
    broken = [code for code in UI_LANGUAGE_CODES
              if ui_language_code(ui_language_name(code)) != code]
    check("подпись и код языка интерфейса сходятся", not broken,
          ", ".join(ui_language_name(c) for c in UI_LANGUAGE_CODES))

    # Метки каналов переводятся, а чужие метки проходят насквозь — на этом
    # держатся и selftest, и любая своя метка в Config.
    was = i18n.get_language()
    try:
        i18n.set_language("en")
        english = (speaker("me"), speaker("them"))
        i18n.set_language("ru")
        russian = (speaker("me"), speaker("them"))
    finally:
        i18n.set_language(was)
    check("метки говорящих переводятся",
          english == ("Me", "Them") and russian == ("Я", "Собеседник"),
          f"{english} / {russian}")
    check("незнакомая метка возвращается как есть", speaker("Тест") == "Тест")


def test_settings() -> None:
    """Сохранённый выбор: переживает перезапуск и не мешает старту."""
    section("Сохранение настроек")
    import json
    import tempfile
    from pathlib import Path

    from callcribe.config import CFG
    from callcribe.settings import (
        MAX_CUSTOM_MODELS,
        Settings,
        SettingsStore,
        model_display,
        model_labels,
        model_problem,
    )

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)

        # Первый запуск: файла нет — это не проблема, а обычное дело.
        store = SettingsStore(root / "settings.json")
        problems = store.load()
        check("без файла настроек — ни одной жалобы", not problems,
              f"жалоб: {len(problems)}")
        check("значения по умолчанию берутся из Config",
              store.data.whisper_model == CFG.whisper_model
              and store.data.language == CFG.language,
              f"{store.data.whisper_model}, язык {store.data.language!r}")

        # Круг: записали -> прочитали другим объектом.
        store.data.ui_language = "ru"
        store.data.language = None          # «Авто»
        store.data.whisper_model = "large-v3-turbo"
        store.data.custom_models = [str(root / "my-model")]
        check("сохранение проходит без ошибок", store.save() is None)
        check("временный файл за собой не оставлен",
              not list(root.glob("*.tmp")), [p.name for p in root.glob("*")])

        again = SettingsStore(root / "settings.json")
        problems = again.load()
        check("выбор пережил перезапуск",
              (again.data.ui_language, again.data.whisper_model) == ("ru", "large-v3-turbo"),
              f"{again.data.ui_language}, {again.data.whisper_model}")
        # None здесь — полноправное значение «Авто», а не «не задано».
        # Спутать их значит терять выбор «Авто» при каждом запуске.
        check("«Авто» сохраняется как выбор, а не как пустота",
              again.data.language is None and not problems,
              f"язык {again.data.language!r}, жалоб {len(problems)}")

        # Испорченный файл не должен мешать запуску.
        broken = root / "broken.json"
        broken.write_text("{это не json", encoding="utf-8")
        store = SettingsStore(broken)
        problems = store.load()
        check("испорченный файл — предупреждение, а не отказ",
              len(problems) == 1 and problems[0][0] == "settings.unreadable",
              problems[0][0] if problems else "промолчали")
        check("после испорченного файла берутся умолчания",
              store.data.whisper_model == CFG.whisper_model)

        # Файл правильный, но не тот: чужой JSON вместо словаря настроек.
        alien = root / "alien.json"
        alien.write_text("[1, 2, 3]", encoding="utf-8")
        store = SettingsStore(alien)
        check("чужой JSON тоже не роняет старт",
              [key for key, _ in store.load()] == ["settings.unreadable"])

        # Недопустимые значения полей — жалоба на каждое, а не отказ.
        odd = root / "odd.json"
        odd.write_text(json.dumps({
            "ui_language": "fr", "language": "de",
            "whisper_model": "", "custom_models": "не список",
        }), encoding="utf-8")
        store = SettingsStore(odd)
        keys = [key for key, _ in store.load()]
        check("на каждое негодное поле — своя жалоба",
              keys == ["settings.bad_value"] * 4, f"жалоб: {len(keys)}")
        check("негодные поля заменены умолчаниями",
              (store.data.ui_language, store.data.language, store.data.whisper_model)
              == (CFG.ui_language, CFG.language, CFG.whisper_model),
              f"{store.data.ui_language}, {store.data.language!r}")

        # Папку с моделью могли удалить или отключить диск — выбор нужно
        # снять сразу, а не ловить отказ загрузки через полминуты.
        gone = root / "gone.json"
        gone.write_text(json.dumps({"whisper_model": str(root / "нет-такой-папки")}),
                        encoding="utf-8")
        store = SettingsStore(gone)
        keys = [key for key, _ in store.load()]
        check("исчезнувшая папка модели замечена при старте",
              keys == ["settings.model_gone"], f"жалобы: {keys}")
        check("вместо неё встаёт модель по умолчанию",
              store.data.whisper_model == CFG.whisper_model, store.data.whisper_model)

        # Сохранить не удалось — это сообщение, а не исключение: уронить
        # окно из-за настройки нельзя.
        blocker = root / "blocker"
        blocker.write_text("я файл, а не папка", encoding="utf-8")
        store = SettingsStore(blocker / "settings.json")
        problem = store.save()
        check("несохранимый путь — сообщение, а не исключение",
              problem is not None and problem[0] == "settings.save_failed",
              problem[0] if problem else "сохранилось?!")

        # --- проверка папки с моделью -----------------------------------
        good = root / "faster-whisper-large-v3"
        good.mkdir()
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (good / name).write_text("x", encoding="utf-8")
        check("годная папка модели принимается", model_problem(str(good)) is None)
        check("имя размера не проверяется как папка",
              model_problem("large-v3") is None)

        no_bin = root / "no-bin"
        no_bin.mkdir()
        (no_bin / "config.json").write_text("x", encoding="utf-8")
        (no_bin / "tokenizer.json").write_text("x", encoding="utf-8")
        problem = model_problem(str(no_bin))
        check("папка без model.bin отвергнута",
              problem is not None and problem[0] == "model.missing_files",
              problem[0] if problem else "принята")

        # Без токенизатора faster-whisper молча полезет за ним на
        # Hugging Face — в офлайн-приложении это зависание, а не отказ.
        no_tok = root / "no-tokenizer"
        no_tok.mkdir()
        for name in ("model.bin", "config.json"):
            (no_tok / name).write_text("x", encoding="utf-8")
        problem = model_problem(str(no_tok))
        check("папка без токенизатора отвергнута",
              problem is not None and problem[0] == "model.no_tokenizer",
              problem[0] if problem else "принята")

        problem = model_problem(str(root / "и-вовсе-нет"))
        check("несуществующий путь отвергнут",
              problem is not None and problem[0] == "model.not_a_dir",
              problem[0] if problem else "принят")

        # --- подписи для выпадающего списка ------------------------------
        check("имя размера остаётся именем размера",
              model_display("large-v3") == "large-v3")
        check("от папки в списке остаётся её имя",
              model_display(str(good)) == "faster-whisper-large-v3",
              model_display(str(good)))

        # Две сборки с одинаковым именем папки в разных каталогах — и обе
        # должны быть выбираемы, иначе список вернёт первую вместо второй.
        twins = [str(root / "a" / "large-v3"), str(root / "b" / "large-v3")]
        labels = model_labels(["large-v3", *twins])
        check("совпавшие имена папок различимы в списке",
              len(labels) == 3 and set(labels.values()) == {"large-v3", *twins},
              " | ".join(labels))

        # --- история выбранных папок --------------------------------------
        data = Settings()
        for index in range(MAX_CUSTOM_MODELS + 3):
            data.remember_model(str(root / f"model-{index}"))
        check("история папок не растёт без предела",
              len(data.custom_models) == MAX_CUSTOM_MODELS,
              f"запомнено {len(data.custom_models)} из {MAX_CUSTOM_MODELS + 3}")
        check("последняя выбранная папка — первая в списке",
              data.custom_models[0] == str(root / f"model-{MAX_CUSTOM_MODELS + 2}"))

        data.remember_model(data.custom_models[-1])
        check("повторный выбор не заводит второй записи",
              len(set(data.custom_models)) == len(data.custom_models),
              f"записей {len(data.custom_models)}")
        data.remember_model("large-v3")
        check("имя размера в историю папок не попадает",
              "large-v3" not in data.custom_models)


class _FakeCapture:
    """Заглушка AudioCapture: VadSegmenter'у нужны только label и out_queue."""

    def __init__(self, label: str):
        self.label = label
        self.out_queue: "queue.Queue[tuple[float, np.ndarray]]" = queue.Queue()


def _run_segmenter(
    cfg: Config, stream: np.ndarray, t0: float, settle: float = 1.5
) -> list[Utterance]:
    """Прогоняет сигнал через сегментатор блоками по read_ms.

    Блоки скармливаются мгновенно, но метки времени проставляются так,
    будто звук шёл в реальном времени начиная с t0 — ровно то, что делает
    поток захвата.
    """
    from callcribe.vad import VadSegmenter

    sr = cfg.sample_rate_target
    size = int(sr * cfg.read_ms / 1000)

    capture = _FakeCapture("Тест")
    out_q: "queue.Queue[Utterance]" = queue.Queue()
    stop = threading.Event()
    pause = threading.Event()

    segmenter = VadSegmenter(capture, cfg, out_q, stop, pause)
    segmenter.start()
    for offset in range(0, len(stream), size):
        block = stream[offset : offset + size]
        capture.out_queue.put((t0 + (offset + len(block)) / sr, block))
    time.sleep(settle)
    stop.set()
    segmenter.join(timeout=5)

    results = []
    while True:
        try:
            results.append(out_q.get_nowait())
        except queue.Empty:
            break
    return results


def test_vad() -> None:
    section("VAD-сегментация")
    cfg = Config(vad_aggressiveness=0)
    sr = cfg.sample_rate_target
    t0 = time.time()

    # 0.5 с тишины, 1.5 с речи, 1.2 с тишины, 1.0 с речи -> две фразы.
    silence = np.zeros(int(1.2 * sr), dtype=np.float32)
    stream = np.concatenate(
        [np.zeros(int(0.5 * sr), np.float32), voiced_signal(1.5, sr), silence,
         voiced_signal(1.0, sr), silence]
    )
    utterances = _run_segmenter(cfg, stream, t0)

    check("две фразы разделены паузой", len(utterances) == 2, f"получено {len(utterances)}")
    if len(utterances) == 2:
        first, second = utterances
        check(
            "длительность первой фразы ~1.5-2.2 с",
            1.4 <= first.duration <= 2.3,
            f"{first.duration:.2f} с",
        )
        # Речь начинается на 0.5 с, преролл 0.3 с -> ждём ~0.2 с от t0.
        check(
            "метка времени — начало фразы, а не конец",
            abs((first.ts - t0) - 0.2) < 0.35,
            f"начало на {first.ts - t0:.2f} с (ожидалось ~0.20)",
        )
        # Вторая фраза стартует на 0.5+1.5+1.2 = 3.2 с, минус преролл -> ~2.9.
        check(
            "промежуток между фразами сохранён",
            2.3 < (second.ts - first.ts) < 3.3,
            f"дельта {second.ts - first.ts:.2f} с (ожидалось ~2.70)",
        )

    # Обрезка хвостовой тишины: сравниваем с конфигом, где не обрезаем.
    untrimmed_cfg = Config(vad_aggressiveness=0, trailing_silence_keep_ms=cfg.end_silence_ms)
    untrimmed = _run_segmenter(untrimmed_cfg, stream, t0)
    if utterances and untrimmed:
        saved = untrimmed[0].duration - utterances[0].duration
        check(
            "хвостовая тишина обрезается",
            saved > 0.3,
            f"короче на {saved:.2f} с ({untrimmed[0].duration:.2f} -> {utterances[0].duration:.2f})",
        )

    # Длинная речь должна резаться по естественным паузам, а не по таймеру.
    # Три "предложения" по 4 с, разделённые паузами в 400 мс: пауза короче
    # end_silence (600 мс), поэтому фраза не считается законченной, но
    # достаточна как место разреза. Именно 400 мс, а не меньше: webrtcvad
    # сглаживает короткие провалы, и на синтетике 250 мс тишины он
    # отмечает лишь 2 кадрами — ниже порога разреза.
    gap = np.zeros(int(0.40 * sr), dtype=np.float32)
    monologue = np.concatenate(
        [voiced_signal(4.0, sr), gap, voiced_signal(4.0, sr), gap,
         voiced_signal(4.0, sr), np.zeros(int(1.5 * sr), np.float32)]
    )
    smart = _run_segmenter(Config(vad_aggressiveness=0, soft_utterance_ms=8_000), monologue, t0)
    blunt = _run_segmenter(Config(vad_aggressiveness=0, soft_utterance_ms=10**9), monologue, t0)

    check(
        "без нарезки по паузам это одна длинная фраза",
        len(blunt) == 1,
        f"фраз: {len(blunt)}, длина {blunt[0].duration:.1f} с" if blunt else "пусто",
    )
    check(
        "нарезка по паузам делит длинную речь",
        len(smart) > len(blunt),
        f"{len(blunt)} -> {len(smart)} фраз ({', '.join(f'{u.duration:.1f}с' for u in smart)})",
    )
    if smart and blunt:
        # Резать по паузам можно только не теряя речь.
        lost = sum(u.duration for u in blunt) - sum(u.duration for u in smart)
        check("речь при этом не теряется", abs(lost) < 1.0, f"расхождение {lost:+.2f} с")
        check(
            "куски укладываются в мягкий предел",
            max(u.duration for u in smart) < 11.0,
            f"самый длинный {max(u.duration for u in smart):.1f} с",
        )

    # Монолог без единой паузы должен нарезаться принудительно.
    long_cfg = Config(vad_aggressiveness=0, max_utterance_ms=2_000, split_overlap_ms=200,
                      soft_utterance_ms=10**9)
    long_stream = np.concatenate([voiced_signal(5.0, sr), np.zeros(int(1.0 * sr), np.float32)])
    pieces = _run_segmenter(long_cfg, long_stream, t0)
    check("длинный монолог нарезан на куски", len(pieces) >= 2, f"кусков: {len(pieces)}")
    if len(pieces) >= 2:
        ordered = all(pieces[i].ts < pieces[i + 1].ts for i in range(len(pieces) - 1))
        check("время кусков строго возрастает", ordered,
              " -> ".join(f"{p.ts - t0:.2f}" for p in pieces))


def _voiced_frame_counts(utterances, frame_len: int) -> Counter:
    """Сколько раз каждый звучащий кадр попал в отданные фразы.

    Кадры тишины не считаем: они побитово одинаковы (нули), и одинаковость
    в них ничего не означает. Интересен только звук.
    """
    counts: Counter = Counter()
    for utt in utterances:
        for offset in range(0, len(utt.audio) - frame_len + 1, frame_len):
            frame = utt.audio[offset : offset + frame_len]
            if float(np.max(np.abs(frame))) > 1e-4:
                counts[frame.tobytes()] += 1
    return counts


def _duplicated_frames(utterances, frame_len: int) -> int:
    return sum(n - 1 for n in _voiced_frame_counts(utterances, frame_len).values() if n > 1)


def _worst_time_overlap(utterances) -> float:
    """Насколько соседние фразы одного канала налезают друг на друга, в мс.

    Отрицательное значение — между ними промежуток, это норма.
    """
    worst = -1e9
    ordered = sorted(utterances, key=lambda u: u.ts)
    for prev, nxt in zip(ordered, ordered[1:]):
        worst = max(worst, (prev.ts + prev.duration - nxt.ts) * 1000)
    return worst


def test_duplicates() -> None:
    """Повторы в расшифровке — то, что видно глазом как «текст задвоился».

    Механизма три, и путать их нельзя:
      1. сегментатор отдал один и тот же звук дважды — это был бы наш баг;
      2. перехлёст при жёсткой нарезке — наш умысел, но он обязан быть
         ровно тем, который заказан в split_overlap_ms;
      3. эхо колонок в микрофон — фраза приходит по обоим каналам сразу;
         это не сбой конвейера, а работа без наушников.
    """
    section("Повторы в расшифровке")
    from callcribe.vad import VadSegmenter

    sr = Config().sample_rate_target
    t0 = time.time()

    # --- 1. Обычная речь: ни один кадр не уходит дважды ------------------
    cfg = Config(vad_aggressiveness=0)
    two_phrases = np.concatenate([
        np.zeros(int(0.5 * sr), np.float32),
        chirp(1.5, sr, 100.0, 170.0),
        np.zeros(int(1.2 * sr), np.float32),
        chirp(1.0, sr, 200.0, 260.0),
        np.zeros(int(1.2 * sr), np.float32),
    ])
    utterances = _run_segmenter(cfg, two_phrases, t0)
    check("две фразы через паузу — звук не задвоился",
          _duplicated_frames(utterances, cfg.frame_len) == 0,
          f"фраз: {len(utterances)}, повторных кадров: "
          f"{_duplicated_frames(utterances, cfg.frame_len)}")

    # --- 2. Нарезка длинной речи по паузам -------------------------------
    gap = np.zeros(int(0.40 * sr), dtype=np.float32)
    monologue = np.concatenate([
        chirp(4.0, sr, 100.0, 150.0), gap,
        chirp(4.0, sr, 160.0, 210.0), gap,
        chirp(4.0, sr, 220.0, 270.0), np.zeros(int(1.5 * sr), np.float32),
    ])
    soft = _run_segmenter(Config(vad_aggressiveness=0, soft_utterance_ms=8_000), monologue, t0)
    dups = _duplicated_frames(soft, cfg.frame_len)
    check("нарезка по паузам не дублирует звук", dups == 0,
          f"кусков: {len(soft)}, повторных кадров: {dups}")
    check("куски по времени не налезают друг на друга",
          _worst_time_overlap(soft) <= 1.0,
          f"худший перехлёст {_worst_time_overlap(soft):+.0f} мс")

    # --- 3. Жёсткая нарезка: перехлёст ровно заказанный ------------------
    # Речь без единой паузы (сплошной чирп) — режется по таймеру, и вот
    # ЗДЕСЬ повтор запланирован: иначе слово на стыке пропадёт целиком.
    seamless = np.concatenate([chirp(5.0, sr, 100.0, 260.0),
                               np.zeros(int(1.0 * sr), np.float32)])
    hard_cfg = Config(vad_aggressiveness=0, max_utterance_ms=2_000,
                      split_overlap_ms=200, soft_utterance_ms=10**9)
    hard = _run_segmenter(hard_cfg, seamless, t0)
    dups = _duplicated_frames(hard, hard_cfg.frame_len)
    budget = max(0, len(hard) - 1) * hard_cfg.overlap_frames
    check("перехлёст при жёсткой нарезке не больше заказанного",
          len(hard) >= 2 and 0 < dups <= budget,
          f"кусков: {len(hard)}, повторных кадров: {dups}, предел {budget}")
    check("перехлёст виден и по времени",
          0 < _worst_time_overlap(hard) <= hard_cfg.split_overlap_ms + 1,
          f"{_worst_time_overlap(hard):+.0f} мс при заказанных "
          f"{hard_cfg.split_overlap_ms}")

    # Тот же поток без перехлёста — повторов не должно остаться совсем.
    no_overlap_cfg = Config(vad_aggressiveness=0, max_utterance_ms=2_000,
                            split_overlap_ms=0, soft_utterance_ms=10**9)
    plain = _run_segmenter(no_overlap_cfg, seamless, t0)
    dups = _duplicated_frames(plain, no_overlap_cfg.frame_len)
    check("при split_overlap_ms=0 повторов нет вовсе", dups == 0,
          f"кусков: {len(plain)}, повторных кадров: {dups}")

    # --- 4. Зацикливание декодера ----------------------------------------
    # Whisper повторяет фразу по кругу; на двух-трёх повторах сжатие до
    # порога не дотягивает, ловить обязан детектор оборотов.
    looped = [
        "Надо править контекст Entity Framework. Надо править контекст Entity Framework.",
        "Миграция базы данных не накатилась. Миграция базы данных не накатилась.",
        "Давай вынесем это в middleware. Давай вынесем это в middleware. "
        "Давай вынесем это в middleware.",
        "Продолжение следует. Продолжение следует. Продолжение следует. Продолжение следует.",
    ]
    for text in looped:
        check(f"отсеян повтор фразы: {text[:38]!r}...", filters.is_hallucination(text))

    # Настоящая речь, которая ЛЕГАЛЬНО повторяется. Первая строка — из
    # реальной расшифровки: человек считал вслух.
    genuine = [
        "Раз, два, три. Раз, два, три.",
        "Да, конечно, конечно.",
        "Нет, нет, я про другое",
        "It's burning, it's burning",
    ]
    for text in genuine:
        check(f"живой повтор сохранён: {text!r}", not filters.is_hallucination(text))

    # --- 5. Эхо между каналами -------------------------------------------
    # Работа без наушников: голос из колонок уходит в loopback И попадает
    # в микрофон. Конвейер обязан отдать ДВЕ фразы — склеить их он не может
    # и не должен, — но по метке канала их видно, что это одно и то же.
    from callcribe.models import Line
    from callcribe.transcript import TranscriptWriter

    phrase = np.concatenate([np.zeros(int(0.3 * sr), np.float32),
                             chirp(1.5, sr, 120.0, 180.0),
                             np.zeros(int(1.2 * sr), np.float32)])
    echo_q: "queue.Queue[Utterance]" = queue.Queue()
    for label, delay in (("them", 0.0), ("me", 0.12)):
        capture = _FakeCapture(label)
        stop, pause = threading.Event(), threading.Event()
        segmenter = VadSegmenter(capture, cfg, echo_q, stop, pause)
        segmenter.start()
        size = int(sr * cfg.read_ms / 1000)
        for offset in range(0, len(phrase), size):
            block = phrase[offset : offset + size]
            capture.out_queue.put((t0 + delay + (offset + len(block)) / sr, block))
        time.sleep(1.5)
        stop.set()
        segmenter.join(timeout=5)

    echoed = []
    while True:
        try:
            echoed.append(echo_q.get_nowait())
        except queue.Empty:
            break
    labels = sorted(utt.label for utt in echoed)
    check("эхо даёт по фразе на каждый канал, а не одну",
          labels == ["me", "them"], f"метки: {labels}")

    # И вот почему это выглядит как баг: при show_speaker_labels=False
    # обе строки в файле неразличимы.
    same = [Line(t0, "them", "Одно и то же"), Line(t0 + 0.12, "me", "Одно и то же")]
    hidden = TranscriptWriter(Config(show_speaker_labels=False))
    shown = TranscriptWriter(Config(show_speaker_labels=True))
    check(
        "без меток говорящих эхо неотличимо от сбоя",
        len({hidden._format(line)[10:] for line in same}) == 1
        and len({shown._format(line)[10:] for line in same}) == 2,
        "с метками — две разные строки, без меток — две одинаковых",
    )


def test_devices() -> None:
    section("Аудиоустройства")
    try:
        import pyaudiowpatch as pyaudio

        from callcribe.audio import describe, get_loopback_device, get_mic_device
    except ImportError as exc:
        check("pyaudiowpatch установлен", False, str(exc))
        return

    pa = pyaudio.PyAudio()
    try:
        # Loopback обязателен, микрофон — нет: без него приложение
        # запустится и запишет только собеседника.
        try:
            loop = get_loopback_device(pa)
            check("loopback найден", True, describe(loop))
        except Exception as exc:
            check("loopback найден", False, str(exc))

        try:
            mic = get_mic_device(pa)
            check("микрофон найден", True, describe(mic))
        except Exception as exc:
            warn("микрофона нет", str(exc).splitlines()[0])
            print("     -> ваша реплика в расшифровку не попадёт, звонок пишется в одну сторону")
    finally:
        pa.terminate()


def test_dual_capture() -> None:
    """Оба источника одновременно — путь, который дольше всего был не проверен.

    Ровно здесь приложение и падало, когда к машине появился микрофон:
    два потока входили в Pa_Initialize() разом (access violation), а после
    замка второй оставался без COM-апартамента и получал -9999. С одним
    источником оба отказа невидимы, так что тест обязан быть именно на двух.
    """
    section("Два источника одновременно")
    import pyaudiowpatch as pyaudio

    from callcribe.audio import AudioCapture, get_loopback_device, get_mic_device
    from callcribe.status import INFO, Notifier

    cfg = Config()
    pa = pyaudio.PyAudio()
    try:
        try:
            devices = [(get_mic_device(pa), "Я"), (get_loopback_device(pa), "Собеседник")]
        except Exception as exc:
            warn("нужны оба устройства", str(exc).splitlines()[0])
            return
    finally:
        pa.terminate()

    notifier = Notifier()
    stop, pause = threading.Event(), threading.Event()
    captures = [AudioCapture(dev, label, cfg, stop, pause, notifier) for dev, label in devices]

    try:
        for capture in captures:
            capture.start()
        for capture in captures:
            capture.started.wait(timeout=15)

        problems = []
        while True:
            try:
                notice = notifier.queue.get_nowait()
            except queue.Empty:
                break
            if notice.level != INFO:
                problems.append(notice.text)
        check("оба потока захвата открылись", not problems, "; ".join(problems) or "без жалоб")

        # Идут ли блоки — проверка не строгая, а справочная. На loopback
        # молчащей системы их не будет вовсе, а Bluetooth-микрофон поднимает
        # линию не мгновенно: те же наушники за 1.5 с то отдают 14 блоков,
        # то ни одного. Отказ здесь означал бы «тест мигает», а не «сломано»;
        # регрессию ловит проверка выше, на открытие потоков.
        deadline = time.monotonic() + 6.0
        while time.monotonic() < deadline:
            if all(c.out_queue.qsize() for c in captures):
                break
            time.sleep(0.2)
        for capture in captures:
            got = capture.out_queue.qsize()
            if got:
                print(f"  [ИНФО] [{capture.label}] звук идёт — блоков: {got}")
            elif capture.label == "Я":
                warn(f"[{capture.label}] блоков нет за 6 с",
                     "микрофон спит или занят — ваша реплика может не попасть в расшифровку")
            else:
                print(f"  [ИНФО] [{capture.label}] блоков нет — в системе тишина, это нормально")
    finally:
        stop.set()
        for capture in captures:
            capture.join(timeout=10)
        alive = [c.name for c in captures if c.is_alive()]
        check("потоки захвата завершились", not alive, ", ".join(alive) or "оба остановлены")


def test_cuda() -> None:
    section("CUDA")
    from callcribe.cuda import (
        cuda_device_count,
        missing_cuda_libraries,
        prepare_cuda_dll_path,
    )

    dirs = prepare_cuda_dll_path()
    print(f"  каталогов с CUDA-DLL в пути: {len(dirs)}")
    for directory in dirs:
        print(f"    {directory}")

    count = cuda_device_count()
    if count == 0:
        # Не провал: CPU — поддерживаемая конфигурация, приложение само
        # возьмёт модель полегче. Провалом это было бы только для того,
        # кто GPU ожидал, а таким и адресовано напоминание про колёса.
        warn("CUDA-устройства нет — whisper пойдёт на CPU",
             "рабочая конфигурация; для GPU: pip install -r requirements-gpu.txt")
        return

    check("CUDA-устройство доступно для CTranslate2", True, f"устройств: {count}")

    # Самая коварная поломка из всех: nvcuda.dll идёт с драйвером, поэтому
    # устройство видно всегда, а cuBLAS и cuDNN приезжают отдельными
    # колёсами. Без них модель загружается на cuda молча и падает КАЖДАЯ
    # фраза. Приложение это переживёт (уйдёт на CPU), но видеокарта будет
    # простаивать зря, и человек об этом знать не будет.
    missing = missing_cuda_libraries()
    if missing:
        warn("библиотек счёта CUDA нет — видеокарта простаивает",
             "не хватает: " + ", ".join(missing))
        print("     -> pip install -r requirements-gpu.txt")
        print("     -> без них распознавание уйдёт на CPU, хотя карта видна")
    else:
        check("библиотеки счёта CUDA загружаются", True, "cuBLAS и cuDNN на месте")


def test_model_load() -> None:
    section("Загрузка whisper и распознавание")
    from callcribe.asr import TranscriberWorker
    from callcribe.models import Line
    from callcribe.status import INFO, Notifier
    from callcribe.transcript import TranscriptWriter

    cfg = Config()
    sr = cfg.sample_rate_target
    in_q: "queue.Queue[Utterance]" = queue.Queue()
    out_q: "queue.Queue[Line]" = queue.Queue()
    stop = threading.Event()
    notifier = Notifier()
    writer = TranscriptWriter(cfg)

    worker = TranscriberWorker(cfg, in_q, out_q, stop, notifier, writer)
    worker.start()
    started = time.monotonic()
    worker.ready.wait(timeout=900)
    loaded = worker.ready.is_set() and not worker.failed.is_set()
    check(
        "модель загрузилась",
        loaded,
        f"{worker.device_label} за {time.monotonic() - started:.1f} с",
    )

    def drain_problems() -> list[str]:
        problems = []
        while True:
            try:
                notice = notifier.queue.get_nowait()
            except queue.Empty:
                return problems
            if notice.level != INFO:
                problems.append(notice.text)

    if loaded:
        # Модель трогаем ТОЛЬКО через очередь. Обратиться к worker.model из
        # главного потока нельзя: модель CTranslate2 привязана к потоку,
        # который её создал, и параллельное обращение роняет процесс на
        # выходе (0xC0000409) уже после того, как всё отработало.
        drain_problems()
        began = time.monotonic()
        in_q.put(Utterance("Тест", time.time(), voiced_signal(5.0, sr)))
        deadline = time.monotonic() + 180
        while not in_q.empty() and time.monotonic() < deadline:
            time.sleep(0.2)
        time.sleep(1.5)
        elapsed = time.monotonic() - began

        # Главная проверка: загрузка на cuda проходит и без cuBLAS — он
        # подтягивается лениво, и падает только первое реальное
        # распознавание. Отсутствие жалоб = вычисление реально прошло.
        problems = drain_problems()
        check(
            "вычисление прошло без ошибок (CUDA-ядра доступны)",
            not problems,
            "; ".join(problems) or f"5 с звука за {elapsed:.2f} с",
        )
        # Гармонический сигнал — не речь; правильный результат ПУСТО,
        # а не выдуманная фраза.
        check("на не-речи ничего не выдумано", out_q.empty(), f"строк: {out_q.qsize()}")

        # «Авто» — отдельный путь до модели: язык определяется по фразе,
        # и промах детектора правится подстановкой ближайшего из списка.
        # Гоняем тоже через очередь: модель принадлежит своему потоку.
        worker.language.set(None)
        in_q.put(Utterance("Тест", time.time(), voiced_signal(5.0, sr)))
        deadline = time.monotonic() + 180
        while not in_q.empty() and time.monotonic() < deadline:
            time.sleep(0.2)
        time.sleep(1.5)
        # Жалоба про «Авто» — это и есть штатная правка промаха, а не отказ.
        # Язык интерфейса тут любой, поэтому смотрим оба написания.
        auto_problems = [
            p for p in drain_problems() if "Авто" not in p and "Auto" not in p
        ]
        check(
            "режим «Авто» проходит фразу целиком",
            not auto_problems,
            "; ".join(auto_problems) or "определение языка отработало",
        )

    stop.set()
    worker.join(timeout=30)


def main() -> int:
    use_utf8_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-model", action="store_true", help="загрузить whisper (долго)")
    args = parser.parse_args()

    print(f"CallCribe selftest · Python {sys.version.split()[0]} · {sys.platform}")

    test_resampler()
    test_filters()
    test_i18n()
    test_settings()
    test_languages()
    test_language_routing()
    test_model_switch()
    test_vad()
    test_duplicates()
    test_devices()
    test_dual_capture()
    test_cuda()
    if args.load_model:
        test_model_load()

    summary = f"\nИтого: {PASSED} пройдено, {FAILED} провалено"
    if WARNED:
        summary += f", {WARNED} предупреждение(й)"
    print(summary)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
