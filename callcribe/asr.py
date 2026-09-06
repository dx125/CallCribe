"""Распознавание речи. Один поток на весь процесс — модель грузится
один раз, очередь фраз естественным образом сериализует доступ к ней.

ВАЖНО: self.model принадлежит этому потоку и только ему. Модель
CTranslate2 не потокобезопасна, и обращение к ней извне не просто
портит результат — процесс падает с 0xC0000409 при завершении, уже
после того как всё отработало и вывод выглядит корректным. Никогда не
вызывайте worker.model напрямую, отправляйте фразы в очередь.

По той же причине смена модели из окна сделана заявкой (ModelSetting), а
не присваиванием: и создание, и уничтожение модели должны случиться
здесь же, в этом потоке.
"""

from __future__ import annotations

import gc
import queue
import threading
import time

from . import filters
from .config import (
    KNOWN_LANGUAGES,
    Config,
    LanguageSetting,
    ModelSetting,
    language_name,
)
from .cuda import cuda_device_count, missing_cuda_libraries, prepare_cuda_dll_path
from .i18n import t
from .models import Line, Utterance
from .settings import is_local_model, model_display
from .status import Notifier
from .transcript import TranscriptWriter

# Глубина очереди, после которой имеет смысл сказать пользователю,
# что распознавание отстаёт от разговора.
_LAG_WARN_ITEMS = 6


class TranscriberWorker(threading.Thread):
    def __init__(
        self,
        cfg: Config,
        transcribe_queue: "queue.Queue[Utterance]",
        gui_queue: "queue.Queue[Line]",
        stop_event: threading.Event,
        notifier: Notifier,
        writer: TranscriptWriter,
        language: LanguageSetting | None = None,
        model_setting: ModelSetting | None = None,
    ):
        super().__init__(daemon=True, name="transcriber")
        self.cfg = cfg
        self.q = transcribe_queue
        self.gui_queue = gui_queue
        self.stop_event = stop_event
        self.notifier = notifier
        self.writer = writer
        # None — не «Авто», а «настройку не передали»: тогда язык берётся из
        # конфигурации и дальше не меняется. Так работает selftest, которому
        # окно не нужно.
        self.language = language if language is not None else LanguageSetting(cfg.language)
        self.model_setting = (
            model_setting if model_setting is not None else ModelSetting(cfg.whisper_model)
        )

        self.ready = threading.Event()
        self.failed = threading.Event()
        # Взведён, пока модель грузится, — и при старте, и при смене на ходу.
        # Окно по нему показывает, что происходит: смена large-v3 занимает
        # десятки секунд, и без этого выглядит как зависание.
        self.loading = threading.Event()
        self.loading_name = ""
        self.device_label = ""
        self.model = None
        self._auto_fix_reported = False

    # ------------------------------------------------------------------

    def _resolve_device(self) -> tuple[str, str]:
        requested = (self.cfg.whisper_device or "auto").lower()
        device = requested
        if device == "auto":
            count = cuda_device_count()
            device = "cuda" if count > 0 else "cpu"
            if count == 0:
                self.notifier.warn(t("asr.no_cuda"))

        if device == "cuda":
            # Видеокарта видна, а библиотек счёта нет — модель загрузится и
            # будет ронять каждую фразу (см. cuda.missing_cuda_libraries).
            # Молча уйти на CPU нельзя: человек должен узнать, что у него
            # простаивает видеокарта и как это чинится одной командой.
            missing = missing_cuda_libraries()
            if missing:
                self.notifier.warn(t(
                    "asr.cuda_libs_missing", libraries=", ".join(missing)
                ))
                # Выбор "cuda" руками оставляем как выбрали: подменять явное
                # решение — не наше дело, а предупреждение уже прозвучало.
                if requested == "auto":
                    device = "cpu"

        compute = self.cfg.whisper_compute or ("float16" if device == "cuda" else "int8")
        return device, compute

    def _build_model(self, model_name: str, cpu_fallback: str | None):
        """Загрузить модель. Возвращает (модель, имя, подпись устройства).

        cpu_fallback — на что заменить модель, если GPU не поднялся. При
        старте это large-v3-turbo: на CPU он единственный успевает за
        разговором. А при смене модели из окна подменять её нельзя — там
        cpu_fallback=None, и на CPU повторяется та же модель, которую
        попросили: пользователь выбрал явно, и молча дать ему другую
        значит соврать в подписи.
        """
        from faster_whisper import WhisperModel  # импорт после настройки путей к DLL

        device, compute = self._resolve_device()

        # Машина без видеокарты. large-v3 здесь загрузится без единой
        # ошибки и будет отставать от разговора навсегда: человек увидит
        # не «медленно», а растущую очередь и пустое окно — и решит, что
        # приложение сломано. Поэтому подменяем модель до загрузки, а не
        # после первой опоздавшей фразы.
        #
        # Только на старте (cpu_fallback задан) и только для имени размера:
        # выбранную вручную папку подменять нечем, да и выбрана она явно.
        if (
            cpu_fallback
            and device == "cpu"
            and model_name != cpu_fallback
            and not is_local_model(model_name)
        ):
            self.notifier.warn(t(
                "asr.cpu_model_swap",
                requested=model_display(model_name),
                model=model_display(cpu_fallback),
            ))
            model_name = cpu_fallback

        try:
            self.notifier.info(
                t("asr.loading", model=model_display(model_name),
                  device=device, compute=compute)
            )
            model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as exc:
            if device != "cuda":
                raise
            # Типовые причины: нет cuDNN 9, нет свободной VRAM (LM Studio),
            # старый драйвер. Ни одна из них не повод падать целиком.
            # Но причина бывает и вовсе не в видеокарте — например, в
            # выбранной папке нет model.bin, — поэтому виноватым здесь
            # никого не назначаем, а просто повторяем на CPU.
            self.notifier.warn(t(
                "asr.gpu_retry",
                model=model_display(model_name),
                error=f"{type(exc).__name__}: {exc}",
            ))
            device, compute = "cpu", self.cfg.whisper_compute or "int8"
            model_name = cpu_fallback or model_name
            self.notifier.info(
                t("asr.loading", model=model_display(model_name),
                  device=device, compute=compute)
            )
            model = WhisperModel(model_name, device=device, compute_type=compute)

        return model, model_name, f"{model_display(model_name)} · {device}/{compute}"

    def _load_model(self):
        found = prepare_cuda_dll_path()
        if found:
            self.notifier.info(t("asr.cuda_dlls", count=len(found)))

        requested = self.model_setting.get()
        self.loading_name = requested
        self.loading.set()
        try:
            model, loaded, label = self._build_model(
                requested, self.cfg.cpu_fallback_model
            )
        finally:
            self.loading.clear()

        self.device_label = label
        self.model_setting.confirm(loaded)
        return model

    def _switch_model(self, requested: str) -> None:
        """Сменить модель по заявке из окна. Вызывается только этим потоком.

        Старую модель отпускаем ДО загрузки новой, а не после: держать в
        памяти две large-v3 сразу — это лишние 3 ГБ, и на видеокарте
        средних размеров переключение просто не влезет. Расплата за это —
        окно, в котором модели нет вообще, поэтому при неудаче сразу же
        поднимаем обратно прежнюю.
        """
        previous = self.model_setting.get()
        self.loading_name = requested
        self.loading.set()
        self.model = None
        gc.collect()

        try:
            model, loaded, label = self._build_model(requested, None)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            try:
                model, loaded, label = self._build_model(previous, None)
            except Exception as restore_exc:
                self.loading.clear()
                self.failed.set()
                self.notifier.fatal(t(
                    "asr.model_restore_failed",
                    model=model_display(requested), error=error,
                    current=model_display(previous),
                    restore_error=f"{type(restore_exc).__name__}: {restore_exc}",
                ))
                return
            self.model, self.device_label = model, label
            self.model_setting.confirm(loaded)
            self.loading.clear()
            self.notifier.warn(t(
                "asr.model_switch_failed",
                model=model_display(requested), error=error,
                current=model_display(previous),
            ))
            return

        self.model, self.device_label = model, label
        self.model_setting.confirm(loaded)
        self.loading.clear()
        self.notifier.info(t("asr.model_switched", device=label))

    def _decode(self, audio, language: str | None):
        return self.model.transcribe(
            audio,
            language=language,
            initial_prompt=self.cfg.prompt_for(language),
            beam_size=self.cfg.beam_size,
            condition_on_previous_text=False,
            # Второй рубеж после webrtcvad: отсекает сегменты, где речи
            # на самом деле нет, до того как декодер начнёт фантазировать.
            vad_filter=self.cfg.use_whisper_vad_filter,
            vad_parameters={
                "threshold": 0.35,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 200,
            },
            no_speech_threshold=self.cfg.no_speech_threshold,
            log_prob_threshold=self.cfg.log_prob_threshold,
            compression_ratio_threshold=self.cfg.compression_ratio_threshold,
        )

    def _nearest_known(self, info) -> str | None:
        """Самый вероятный язык ИЗ НАШЕГО списка по раскладке детектора."""
        probs = getattr(info, "all_language_probs", None) or ()
        allowed = [(prob, code) for code, prob in probs if code in KNOWN_LANGUAGES]
        return max(allowed)[1] if allowed else None

    def _resolve_language(self, audio) -> str | None:
        """Определить язык фразы, ничего не декодируя.

        faster-whisper определяет язык СРАЗУ, а сегменты отдаёт лениво: к
        моменту возврата из transcribe() детектор уже отработал, а декодер
        ещё не начинал. Значит, генератор можно закрыть и позвать модель
        второй раз — с явным языком и, что важнее, с промптом этого языка.
        Лишнего прохода энкодера это не добавляет: в режиме «Авто» детектор
        и так кодирует окно отдельно от декодера.
        """
        segments, info = self._decode(audio, None)
        close = getattr(segments, "close", None)
        if close is not None:
            close()

        detected = info.language
        if detected in KNOWN_LANGUAGES or not self.cfg.restrict_auto_language:
            return detected

        nearest = self._nearest_known(info)
        if nearest is None:            # раскладки нет — доверяем детектору
            return detected

        text = t(
            "asr.auto_off_list",
            detected=detected,
            probability=f"{info.language_probability:.0%}",
            chosen=language_name(nearest),
        )
        if self._auto_fix_reported:
            self.notifier.info(text)
        else:
            self._auto_fix_reported = True
            self.notifier.warn(t("asr.auto_off_list_hint", text=text))
        return nearest

    def _transcribe(self, utt: Utterance) -> str:
        language = self.language.get()
        if language is None:
            language = self._resolve_language(utt.audio)
        segments, _info = self._decode(utt.audio, language)

        parts: list[str] = []
        for segment in segments:
            no_speech = getattr(segment, "no_speech_prob", None)
            if no_speech is not None and no_speech > self.cfg.max_no_speech_prob:
                continue
            avg_logprob = getattr(segment, "avg_logprob", None)
            if avg_logprob is not None and avg_logprob < self.cfg.min_avg_logprob:
                continue

            piece = filters.clean(segment.text)
            if not piece or filters.is_hallucination(piece):
                continue
            parts.append(piece)

        text = filters.clean(" ".join(parts))
        if not text or filters.is_hallucination(text):
            return ""
        return text

    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            self.model = self._load_model()
        except Exception as exc:
            self.failed.set()
            self.ready.set()  # чтобы окно перестало ждать
            self.notifier.fatal(t("asr.load_failed", error=f"{type(exc).__name__}: {exc}"))
            return

        self.notifier.info(t("asr.ready", device=self.device_label))
        self.ready.set()
        lag_reported = False

        while True:
            # Заявку на смену модели забираем ПЕРЕД взятием фразы: иначе
            # переключение ждало бы конца текущей — а её как раз и хотят
            # разобрать уже новой моделью.
            self._apply_pending_model()

            try:
                utt = self.q.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                lag_reported = False
                continue

            if self.failed.is_set():   # модель не поднялась после смены
                continue

            pending = self.q.qsize()
            if pending >= _LAG_WARN_ITEMS and not lag_reported:
                self.notifier.warn(t("asr.lagging", count=pending))
                lag_reported = True

            started = time.monotonic()
            try:
                text = self._transcribe(utt)
            except Exception as exc:
                self.notifier.warn(
                    t("asr.phrase_failed", error=f"{type(exc).__name__}: {exc}")
                )
                continue

            if not text:
                continue

            line = Line(utt.ts, utt.label, text)
            # Сначала на диск, потом в окно: если что-то упадёт при отрисовке,
            # расшифровка всё равно уже сохранена.
            self.writer.append(line)
            self.gui_queue.put(line)

            elapsed = time.monotonic() - started
            if elapsed > utt.duration:
                self.notifier.info(t(
                    "asr.slower_than_realtime",
                    duration=utt.duration, elapsed=elapsed,
                ))

    def _apply_pending_model(self) -> None:
        if self.model_setting.pending() is None:
            return
        # «Гружусь» взводим ДО того, как забрать заявку. Между этими двумя
        # действиями окно успевает опросить состояние, и увидев «заявки
        # больше нет, и загрузка не идёт», решило бы, что переключение
        # провалилось, — хотя оно ещё и не начиналось.
        self.loading.set()
        requested = self.model_setting.take()
        if requested is None:
            self.loading.clear()
            return
        self.loading_name = requested
        try:
            self._switch_model(requested)
        except Exception as exc:
            # На всякий пожарный: сюда попадать нечем, но остаться без
            # модели молча — худший из исходов.
            self.loading.clear()
            self.failed.set()
            self.notifier.fatal(
                t("asr.load_failed", error=f"{type(exc).__name__}: {exc}")
            )
