"""Все настройки приложения в одном месте."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

from .i18n import DEFAULT_UI_LANGUAGE, UI_LANGUAGE_CODES, t, ui_language_name

# Whisper обрезает initial_prompt до 223 токенов (n_text_ctx // 2 - 1) молча.
# Для смеси английских терминов и русских слов это примерно 600-700 символов.
MAX_PROMPT_CHARS = 650

# ---------------------------------------------------------------------------
# ЯЗЫКИ
# ---------------------------------------------------------------------------

AUTO: str | None = None
"""Автоопределение.

Обозначено None, потому что ровно это whisper и понимает как «определи
сам». Отдельной строкой-меткой ("auto") заводить не стали: она бы жила
только внутри приложения, и её всё равно пришлось бы превращать в None
перед каждым вызовом модели.
"""

LANGUAGE_CODES: tuple[str | None, ...] = ("ru", "en", "es", AUTO)
"""Порядок — тот же, что в выпадающем списке окна.

«Авто» стоит последним намеренно: на коротких репликах детектор языка
ошибается, и первым в списке должен быть вариант, который работает
предсказуемо.
"""

_ENDONYMS: dict[str | None, str] = {
    "ru": "Русский",
    "en": "English",
    "es": "Español",
}
"""Названия языков — на них самих, независимо от языка интерфейса.

Так делают все списки языков, и не из вежливости: человек ищет в списке
знакомое начертание, а «Spanish» ему в этом не помогает. Переводится
только «Авто» — это не язык, а режим.
"""

KNOWN_LANGUAGES: frozenset[str] = frozenset(code for code in LANGUAGE_CODES if code)
"""Языки, которые в этом приложении вообще ожидаются — без «Авто».

Нужен не для показа, а для правки промахов автоопределения: whisper
знает под сотню языков, и на секундной реплике охотно выбирает из них
тот, которого в разговоре быть не может (см. restrict_auto_language).
"""


def language_name(code: str | None) -> str:
    """Код -> подпись для окна."""
    if code == AUTO:
        return t("lang.auto")
    return _ENDONYMS.get(code) or str(code)


def language_code(name: str) -> str | None:
    """Подпись из окна -> код. Неизвестное имя означает «Авто»."""
    for code, label in _ENDONYMS.items():
        if label == name:
            return code
    return AUTO


def language_names() -> list[str]:
    """Подписи для выпадающего списка — в порядке LANGUAGE_CODES.

    Собирается каждый раз заново: подпись «Авто» зависит от языка
    интерфейса, а его меняют, не перезапуская приложение.
    """
    return [language_name(code) for code in LANGUAGE_CODES]


# ---------------------------------------------------------------------------
# МОДЕЛИ
# ---------------------------------------------------------------------------

WHISPER_MODELS: tuple[str, ...] = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
    "large-v3-turbo",
    "distil-large-v3",
)
"""Имена, которые faster-whisper скачивает сам. Всё остальное считается
путём к папке с уже переведённой в CTranslate2 моделью — её выбирают
кнопкой «Выбрать папку...», см. settings.model_problem().

Порядок — от лёгких к тяжёлым: в списке ищут глазами компромисс между
скоростью и точностью, а не алфавит.
"""


class ModelSetting:
    """Выбранная модель: значение плюс заявка на смену.

    Смена модели — это не присваивание, а работа на несколько десятков
    секунд, и делать её может ТОЛЬКО поток распознавания: модель
    CTranslate2 принадлежит создавшему её потоку (см. asr.py). Поэтому
    окно кладёт сюда заявку, а поток её забирает между фразами.
    """

    def __init__(self, name: str):
        self._name = name
        self._pending: str | None = None
        self._lock = threading.Lock()

    def get(self) -> str:
        """Что загружено сейчас."""
        with self._lock:
            return self._name

    def pending(self) -> str | None:
        """На что просили сменить, не забирая заявку (для окна)."""
        with self._lock:
            return self._pending

    def request(self, name: str) -> None:
        with self._lock:
            # Заявка «на то же самое» — это отсутствие заявки, иначе
            # выбор уже выбранного затевал бы перезагрузку модели.
            self._pending = None if name == self._name else name

    def take(self) -> str | None:
        """Поток распознавания забирает заявку. None — менять нечего."""
        with self._lock:
            pending, self._pending = self._pending, None
            return pending

    def confirm(self, name: str) -> None:
        """Загрузка удалась — вот что теперь стоит на самом деле."""
        with self._lock:
            self._name = name


class LanguageSetting:
    """Выбранный язык — единственная настройка, которую можно менять на ходу.

    Пишет в неё поток окна, читает поток распознавания, поэтому это
    отдельный объект, а не поле Config: конфигурация раздана по потокам
    как read-only снимок, и заводить в ней одно мутирующее поле — прямой
    путь к тому, что через полгода мутировать начнут все.

    Замок здесь не про гонку за байты (её и так нет: значение подменяется
    целиком), а про явный контракт — у настройки ровно две точки доступа.
    """

    def __init__(self, code: str | None = AUTO):
        self._code = code
        self._lock = threading.Lock()

    def get(self) -> str | None:
        with self._lock:
            return self._code

    def set(self, code: str | None) -> None:
        with self._lock:
            self._code = code


# Подсказка терминологии — своя на каждый язык. Общее ядро (имена
# технологий) везде одинаково: они и на слух звучат одинаково, и страдают
# при распознавании чаще всего. Различается обвязка — те слова, которые
# в каждом языке свои.
_PROMPT_RU = (
    "ASP.NET Core, Identity, IdentityServer, Duende, OpenIddict, "
    "JWT, OAuth2, OpenID Connect, cookie authentication, bearer token, "
    "Entity Framework Core, EF Core, миграция, DbContext, "
    "Blazor, Razor Pages, minimal API, middleware, DI-контейнер, "
    "claims, роли, политики авторизации, refresh token, "
    "xUnit, NSubstitute, Serilog, appsettings.json, Program.cs, "
    "code review, pull request, merge conflict, pipeline, deploy."
)

_PROMPT_EN = (
    "ASP.NET Core, Identity, IdentityServer, Duende, OpenIddict, "
    "JWT, OAuth2, OpenID Connect, cookie authentication, bearer token, "
    "Entity Framework Core, EF Core, migration, DbContext, "
    "Blazor, Razor Pages, minimal API, middleware, dependency injection, "
    "claims, roles, authorization policies, refresh token, "
    "xUnit, NSubstitute, Serilog, appsettings.json, Program.cs, "
    "code review, pull request, merge conflict, pipeline, deploy."
)

_PROMPT_ES = (
    "ASP.NET Core, Identity, IdentityServer, Duende, OpenIddict, "
    "JWT, OAuth2, OpenID Connect, cookie authentication, bearer token, "
    "Entity Framework Core, EF Core, migración, DbContext, "
    "Blazor, Razor Pages, minimal API, middleware, inyección de dependencias, "
    "claims, roles, políticas de autorización, refresh token, "
    "xUnit, NSubstitute, Serilog, appsettings.json, Program.cs, "
    "revisión de código, pull request, merge conflict, pipeline, despliegue."
)

# Для «Авто» — только имена технологий, без слов какого бы то ни было
# естественного языка. Промпт из русских слов тянет вывод в русский даже
# тогда, когда whisper верно распознал английскую речь: промпт подаётся
# декодеру как «предыдущий текст», и модель продолжает его язык.
_PROMPT_AUTO = (
    "ASP.NET Core, Identity, IdentityServer, Duende, OpenIddict, "
    "JWT, OAuth2, OpenID Connect, Entity Framework Core, EF Core, DbContext, "
    "Blazor, Razor Pages, minimal API, middleware, refresh token, claims, "
    "xUnit, NSubstitute, Serilog, appsettings.json, Program.cs, "
    "pull request, merge conflict, pipeline, deploy."
)

DEFAULT_PROMPTS: dict[str | None, str] = {
    "ru": _PROMPT_RU,
    "en": _PROMPT_EN,
    "es": _PROMPT_ES,
    AUTO: _PROMPT_AUTO,
}


@dataclass
class Config:
    # --- аудио ---
    sample_rate_target: int = 16_000
    frame_ms: int = 30            # webrtcvad принимает только 10 / 20 / 30
    read_ms: int = 100            # размер блока чтения с устройства

    # --- VAD-сегментация ---
    vad_aggressiveness: int = 2   # 0..3, выше — меньше ложных срабатываний
    preroll_ms: int = 300         # запас звука перед началом речи
    end_silence_ms: int = 600     # тишина, после которой фраза считается законченной
    trailing_silence_keep_ms: int = 150
    """Сколько хвостовой тишины оставить в сегменте.

    Остальное отбрасывается: длинный хвост тишины — главный провокатор
    галлюцинаций whisper ("Продолжение следует...").
    """
    soft_utterance_ms: int = 8_000
    """После какой длины речи начинать искать естественную паузу для разреза.

    Монолог нельзя копить до конца: whisper всё равно работает окном в 30 с,
    а пользователь не должен ждать текст полминуты. Но и резать по таймеру
    посреди слова незачем — после этого порога сегментатор ищет самый тихий
    момент в буфере и режет по нему.
    """
    split_silence_ms: int = 90
    """Минимальная пауза, которую считаем годной точкой разреза.

    Замерено на живой речи: у webrtcvad паузы распределены двугорбо —
    либо короче ~180 мс (между словами), либо длиннее 600 мс (там фраза
    и так заканчивается). Между этими горбами пусто, поэтому порог в
    180 мс не срабатывал никогда. 90 мс = 3 кадра — это межсловный
    промежуток, резать по нему заметно лучше, чем по таймеру посреди слова.
    """
    min_chunk_ms: int = 2_000       # короче этого куски при нарезке не делаем
    max_utterance_ms: int = 20_000  # жёсткий предел, если пауз так и не случилось
    split_overlap_ms: int = 200     # перехлёст при жёсткой нарезке, чтобы не рвать слово
    min_utterance_ms: int = 300     # короче — не отправляем в ASR вообще

    # --- ASR ---
    whisper_model: str = "large-v3"
    """Модель при первом запуске: имя из WHISPER_MODELS или путь к папке.

    Дальше побеждает выбранная в окне — она сохраняется, см. settings.py.
    Чтобы вернуться к этому значению, удалите settings.json.
    """
    whisper_device: str = "auto"    # "auto" | "cuda" | "cpu"
    whisper_compute: str = ""       # "" -> float16 на cuda, int8 на cpu
    cpu_fallback_model: str = "large-v3-turbo"
    """Модель для отката на CPU.

    Замерено на 16 логических ядрах, int8: на CPU обе модели упираются в
    фиксированную стоимость вызова (энкодер всегда считает окно 30 с), а не
    в длину фразы. large-v3: ~3.1 с на вызов, turbo: ~2.6 с, при этом
    наклон у turbo вшестеро меньше (0.026 против 0.162 с на секунду звука).
    На типичных для разговора репликах в 2-5 секунд large-v3 отстаёт от
    реального времени, turbo — успевает, а по WER на замере они вничью.
    """
    language: str | None = "ru"
    """Язык распознавания при первом запуске. Меняется на ходу из окна,
    см. LanguageSetting, и сохраняется до следующего раза (settings.py).

    AUTO (None) — автоопределение. На коротких фразах ненадёжно, поэтому
    по умолчанию стоит явный язык, см. README.
    """
    restrict_auto_language: bool = True
    """Чинить ли промахи автоопределения, подставляя ближайший язык из списка.

    В режиме «Авто» whisper определяет язык по каждой фразе отдельно, и на
    реплике в секунду-полторы регулярно выбирает из своей сотни языков
    что-нибудь вроде валлийского — после чего не расшифровывает, а
    переводит. Раз список ожидаемых языков известен (KNOWN_LANGUAGES),
    такой промах видно сразу, и стоит его исправление недорого: язык
    faster-whisper определяет ДО того, как начнёт декодировать.
    """
    beam_size: int = 5

    # --- фильтры галлюцинаций ---
    # Первые три уходят внутрь faster-whisper и управляют temperature-fallback,
    # последние два применяются к уже полученным сегментам.
    no_speech_threshold: float = 0.6
    log_prob_threshold: float = -1.0
    compression_ratio_threshold: float = 2.4
    max_no_speech_prob: float = 0.6
    min_avg_logprob: float = -1.0
    use_whisper_vad_filter: bool = True   # второй рубеж поверх webrtcvad

    prompts: dict[str | None, str] = field(default_factory=lambda: dict(DEFAULT_PROMPTS))
    """Подсказка терминологии, своя на каждый язык.

    Промпт нельзя держать одной строкой на всё приложение: список русских
    слов, поданный при английской речи, тянет вывод обратно в русский.
    Держите здесь актуальный стек — на замере это самый действенный
    рычаг точности (WER 7.1% -> 5.9%).
    """

    # --- отображение ---
    ui_language: str = DEFAULT_UI_LANGUAGE
    """Язык интерфейса — надписи в окне и сообщения, и только они.

    С языком распознавания не связан никак: на английском интерфейсе
    разбирают русский звонок ровно так же. Меняется на ходу из окна и
    сохраняется до следующего запуска, см. settings.py.
    """
    show_speaker_labels: bool = False
    window_geometry: str = "760x520"
    font: tuple[str, int] = ("Consolas", 11)

    # --- сохранение ---
    output_dir: Path = field(default_factory=lambda: Path.home() / "call-transcripts")

    # ------------------------------------------------------------------
    # производные величины
    # ------------------------------------------------------------------

    @property
    def frame_len(self) -> int:
        """Сэмплов в одном VAD-кадре."""
        return int(self.sample_rate_target * self.frame_ms / 1000)

    @property
    def preroll_frames(self) -> int:
        return max(1, self.preroll_ms // self.frame_ms)

    @property
    def end_silence_frames(self) -> int:
        return max(1, self.end_silence_ms // self.frame_ms)

    @property
    def keep_silence_frames(self) -> int:
        return max(0, self.trailing_silence_keep_ms // self.frame_ms)

    @property
    def soft_frames(self) -> int:
        return max(1, self.soft_utterance_ms // self.frame_ms)

    @property
    def split_silence_frames(self) -> int:
        return max(1, self.split_silence_ms // self.frame_ms)

    @property
    def min_chunk_frames(self) -> int:
        return max(1, self.min_chunk_ms // self.frame_ms)

    @property
    def max_frames(self) -> int:
        return max(1, self.max_utterance_ms // self.frame_ms)

    @property
    def overlap_frames(self) -> int:
        return max(0, self.split_overlap_ms // self.frame_ms)

    def prompt_for(self, language: str | None) -> str:
        """Промпт под выбранный язык.

        Откат на промпт «Авто» — он единственный не привязан к языку, так
        что подставить его безопаснее, чем русский список при испанской речи.
        """
        if language in self.prompts:
            return self.prompts[language]
        return self.prompts.get(AUTO, "")

    def validate(self) -> list[str]:
        """Возвращает список предупреждений (не ошибок) о настройках."""
        warnings: list[str] = []
        if self.frame_ms not in (10, 20, 30):
            warnings.append(t("cfg.frame_ms", value=self.frame_ms))
        if not 0 <= self.vad_aggressiveness <= 3:
            warnings.append(t("cfg.vad_aggressiveness", value=self.vad_aggressiveness))
        if self.language not in LANGUAGE_CODES:
            warnings.append(t(
                "cfg.language_off_list",
                value=self.language,
                available=", ".join(repr(code) for code in LANGUAGE_CODES),
            ))
        if self.ui_language not in UI_LANGUAGE_CODES:
            warnings.append(t(
                "cfg.ui_language_off_list",
                value=self.ui_language,
                fallback=ui_language_name(DEFAULT_UI_LANGUAGE),
                available=", ".join(repr(code) for code in UI_LANGUAGE_CODES),
            ))
        for code in LANGUAGE_CODES:
            if code not in self.prompts:
                warnings.append(t("cfg.prompt_missing", language=language_name(code)))
                continue
            if len(self.prompts[code]) > MAX_PROMPT_CHARS:
                warnings.append(t(
                    "cfg.prompt_too_long",
                    language=language_name(code),
                    length=len(self.prompts[code]),
                    limit=MAX_PROMPT_CHARS,
                ))
        if self.end_silence_ms < self.frame_ms * 2:
            warnings.append(t("cfg.end_silence_small"))
        if self.max_utterance_ms > 30_000:
            warnings.append(t("cfg.max_utterance_big", value=self.max_utterance_ms))
        if self.split_silence_ms >= self.end_silence_ms:
            warnings.append(t("cfg.split_never"))
        if self.soft_utterance_ms > self.max_utterance_ms:
            warnings.append(t("cfg.soft_over_max"))
        return warnings


CFG = Config()
