"""Язык интерфейса.

Отдельная настройка от языка распознавания: на английском интерфейсе
вполне могут разбирать русский звонок. Поэтому здесь нет и не должно быть
ничего про whisper — только надписи.

Каталог устроен ключами, а не английскими фразами-ключами: при опечатке
в ключе видно сразу (наружу вылезает сам ключ), а `selftest` дополнительно
сверяет, что каждый ключ из кода есть в каталоге, что у каждого ключа есть
оба языка и что подстановки в них совпадают.

Переводчик один на процесс и живёт модулем: язык интерфейса в приложении
ровно один, а сообщения рождаются в четырёх потоках сразу — протаскивать
объект через каждый из них значит переписать все сигнатуры ради значения,
которое всё равно глобально.
"""

from __future__ import annotations

import threading
from string import Formatter

DEFAULT_UI_LANGUAGE = "en"
"""По умолчанию английский — на нём приложение и открывается впервые."""

UI_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("English", "en"),
    ("Русский", "ru"),
)
"""Подписи для выпадающего списка. Названия языков — на них самих:
пользователю, который ищет свой язык, английское "Russian" не помогает.
"""

UI_LANGUAGE_CODES: tuple[str, ...] = tuple(code for _, code in UI_LANGUAGES)


def ui_language_name(code: str) -> str:
    for name, value in UI_LANGUAGES:
        if value == code:
            return name
    return code


def ui_language_code(name: str) -> str:
    for label, value in UI_LANGUAGES:
        if label == name:
            return value
    return DEFAULT_UI_LANGUAGE


# ---------------------------------------------------------------------------
# КАТАЛОГ
# ---------------------------------------------------------------------------

MESSAGES: dict[str, dict[str, str]] = {
    # --- языки распознавания -------------------------------------------
    "lang.auto": {"en": "Auto", "ru": "Авто"},

    # --- метки говорящих ------------------------------------------------
    # Попадают и в окно, и в сохранённый файл, поэтому короткие.
    "label.me": {"en": "Me", "ru": "Я"},
    "label.them": {"en": "Them", "ru": "Собеседник"},

    # --- запуск ---------------------------------------------------------
    "app.audio_error_title": {
        "en": "CallCribe — audio capture error",
        "ru": "CallCribe — ошибка захвата звука",
    },
    "app.no_soxr": {
        "en": "soxr is not installed — resampling falls back to numpy. "
              "It works, but quality is worse: pip install soxr",
        "ru": "soxr не установлен — ресемплинг идёт по numpy-фолбэку. "
              "Работает, но качество хуже: pip install soxr",
    },
    "app.loopback": {
        "en": "system audio (loopback): {device}",
        "ru": "системный звук (loopback): {device}",
    },
    "app.mic": {"en": "microphone: {device}", "ru": "микрофон: {device}"},
    "app.mic_unavailable": {
        "en": "microphone unavailable, recording the other side only. {error}",
        "ru": "микрофон недоступен, пишу только собеседника. {error}",
    },
    "app.no_device": {"en": "no device", "ru": "нет устройства"},
    "app.speech_language": {
        "en": "speech language: {language}",
        "ru": "язык распознавания: {language}",
    },
    "app.interface_language": {
        "en": "interface language: {language}",
        "ru": "язык интерфейса: {language}",
    },
    "app.model": {"en": "model: {model}", "ru": "модель: {model}"},
    "app.saved": {
        "en": "[save] transcript saved: {path}",
        "ru": "[save] расшифровка сохранена: {path}",
    },
    "app.nothing_saved": {
        "en": "[save] nothing to save",
        "ru": "[save] пусто, нечего сохранять",
    },

    # --- аудио ----------------------------------------------------------
    "audio.no_wasapi": {
        "en": "WASAPI is not available on this system. CallCribe only runs on Windows.",
        "ru": "WASAPI недоступен в этой системе. CallCribe работает только на Windows.",
    },
    "audio.no_mic": {
        "en": "There is no default microphone in the system.\n\n"
              "Plug in a headset and check: Settings -> System -> Sound -> Input. "
              "Disabled devices are enabled there too, via «Device properties».",
        "ru": "В системе нет микрофона по умолчанию.\n\n"
              "Подключите гарнитуру и проверьте: Параметры -> Система -> Звук -> "
              "Ввод. Отключённые устройства там же включаются через "
              "«Свойства устройства».",
    },
    "audio.no_output": {
        "en": "No default audio output device found.",
        "ru": "Не найдено устройство вывода звука по умолчанию.",
    },
    "audio.no_loopback": {
        "en": "Could not find a loopback device for the current audio output "
              "({device}).\n\n"
              "Usually this means the output device is held by another application "
              "in exclusive mode. Check: Sound settings -> Device properties -> "
              "Advanced -> clear «Allow applications to take exclusive control of "
              "this device».",
        "ru": "Не нашёл loopback-устройство для текущего вывода звука "
              "({device}).\n\n"
              "Обычно это значит, что устройство вывода занято другим приложением "
              "в эксклюзивном режиме. Проверьте: Параметры звука -> Свойства "
              "устройства -> Дополнительно -> снять «Разрешить приложениям "
              "использовать устройство в монопольном режиме».",
    },
    "audio.describe": {
        "en": "{name} ({rate} Hz, {channels} ch.)",
        "ru": "{name} ({rate} Гц, {channels} кан.)",
    },
    "audio.open_failed": {
        "en": "[{label}] could not open the audio stream: {error}",
        "ru": "[{label}] не удалось открыть аудиопоток: {error}",
    },
    "audio.stopped": {
        "en": "[{label}] capture stopped — the device was unplugged or switched.",
        "ru": "[{label}] захват остановился — устройство отключено или переключено.",
    },
    "audio.process_errors": {
        "en": "[{label}] audio processing errors: {error}",
        "ru": "[{label}] ошибки обработки звука: {error}",
    },

    # --- распознавание ---------------------------------------------------
    "asr.no_cuda": {
        "en": "CUDA not detected — transcription will run on the CPU. "
              "Check that nvidia-cublas-cu12 and nvidia-cudnn-cu12 are installed.",
        "ru": "CUDA не обнаружена — распознавание пойдёт на CPU. "
              "Проверьте, что установлены nvidia-cublas-cu12 и nvidia-cudnn-cu12.",
    },
    # Драйвер и библиотеки счёта — разные вещи, и разойтись они могут
    # запросто; сообщение должно назвать и то, чего не хватает, и команду.
    "asr.cuda_libs_missing": {
        "en": "CUDA compute libraries are missing ({libraries}): the GPU driver is "
              "there, but every phrase would fail on them. Fix with: "
              "pip install -r requirements-gpu.txt",
        "ru": "не хватает библиотек счёта CUDA ({libraries}): драйвер видеокарты "
              "есть, но на них падала бы каждая фраза. Чинится так: "
              "pip install -r requirements-gpu.txt",
    },
    "asr.cuda_dlls": {
        "en": "CUDA libraries: {count} folder(s) added to the path",
        "ru": "CUDA-библиотеки: {count} каталог(ов) добавлено в путь",
    },
    "asr.loading": {
        "en": "loading {model} ({device}/{compute})...",
        "ru": "загружаю {model} ({device}/{compute})...",
    },
    # Не «GPU недоступен»: на этой ветке мы ещё не знаем, чья вина.
    # Отказаться загрузиться на видеокарте модель может и потому, что нет
    # cuDNN, и потому, что в выбранной папке просто нет model.bin, — а
    # уверенная жалоба на GPU уводит от настоящей причины.
    "asr.gpu_retry": {
        "en": "{model} did not load on the GPU ({error}). Retrying on the CPU.",
        "ru": "{model} не загрузилась на видеокарте ({error}). Пробую на CPU.",
    },
    # Подмена на старте, а не жалоба задним числом: см. asr._build_model.
    "asr.cpu_model_swap": {
        "en": "no GPU: {requested} would not keep up with a conversation on the CPU, "
              "starting with {model} instead. You can pick another one in the window.",
        "ru": "видеокарты нет: {requested} на CPU не успеет за разговором, "
              "стартую на {model}. В окне можно выбрать другую.",
    },
    "asr.ready": {"en": "ready, listening ({device})", "ru": "готово, слушаю ({device})"},
    "asr.load_failed": {
        "en": "could not load the model: {error}",
        "ru": "не удалось загрузить модель: {error}",
    },
    "asr.lagging": {
        "en": "transcription is falling behind: {count} phrases queued. "
              "If this keeps happening — pick a lighter model (large-v3-turbo) "
              "or switch to the GPU.",
        "ru": "распознавание отстаёт: в очереди {count} фраз. "
              "Если это повторяется — возьмите модель полегче "
              "(large-v3-turbo) или переключитесь на GPU.",
    },
    "asr.phrase_failed": {
        "en": "phrase transcription failed: {error}",
        "ru": "сбой распознавания фразы: {error}",
    },
    "asr.slower_than_realtime": {
        "en": "a {duration:.1f}s phrase took {elapsed:.1f}s — slower than real time",
        "ru": "фраза {duration:.1f}с распознана за {elapsed:.1f}с — "
              "медленнее реального времени",
    },
    "asr.auto_off_list": {
        "en": "«Auto»: phrase language detected as {detected} ({probability}), "
              "it is not on the list — using {chosen}",
        "ru": "«Авто»: язык фразы определён как {detected} ({probability}), "
              "в списке его нет — беру {chosen}",
    },
    "asr.auto_off_list_hint": {
        "en": "{text}. On short phrases this is routine: if you know the call "
              "language, pick it explicitly in the window.",
        "ru": "{text}. На коротких репликах это обычное дело: если язык "
              "звонка известен, выберите его в окне явно.",
    },
    "asr.model_switched": {
        "en": "model switched: {device}",
        "ru": "модель переключена: {device}",
    },
    "asr.model_switch_failed": {
        "en": "could not load {model} ({error}) — staying on {current}",
        "ru": "не удалось загрузить {model} ({error}) — остаюсь на {current}",
    },
    "asr.model_restore_failed": {
        "en": "could not load {model} ({error}), and the previous model {current} "
              "did not come back either ({restore_error}). Transcription has stopped — "
              "restart the application.",
        "ru": "не удалось загрузить {model} ({error}), и прежняя модель {current} "
              "тоже не вернулась ({restore_error}). Распознавание остановлено — "
              "перезапустите приложение.",
    },

    # --- проверка настроек ------------------------------------------------
    "cfg.frame_ms": {
        "en": "frame_ms={value}: webrtcvad only accepts 10/20/30 ms",
        "ru": "frame_ms={value}: webrtcvad принимает только 10/20/30 мс",
    },
    "cfg.vad_aggressiveness": {
        "en": "vad_aggressiveness={value}: allowed range is 0..3",
        "ru": "vad_aggressiveness={value}: допустимо 0..3",
    },
    "cfg.language_off_list": {
        "en": "language={value!r} is not in the window's language list — you will "
              "not be able to switch back to it. Available: {available}",
        "ru": "language={value!r} нет в списке языков окна — переключить "
              "обратно на него будет уже нельзя. Доступны: {available}",
    },
    "cfg.ui_language_off_list": {
        "en": "ui_language={value!r} is unknown, the interface will be in "
              "{fallback}. Available: {available}",
        "ru": "ui_language={value!r} неизвестен, интерфейс будет на "
              "{fallback}. Доступны: {available}",
    },
    "cfg.prompt_missing": {
        "en": "no prompt for {language} in prompts — terminology will not be hinted",
        "ru": "для языка {language} нет промпта в prompts — "
              "термины подсказаны не будут",
    },
    "cfg.prompt_too_long": {
        "en": "the prompt for «{language}» is {length} characters — whisper will "
              "silently cut it to 223 tokens (~{limit} characters). "
              "Drop the less important terms.",
        "ru": "промпт для «{language}» длиной {length} символов — whisper молча "
              "обрежет его до 223 токенов (~{limit} символов). "
              "Уберите менее важные термины.",
    },
    "cfg.end_silence_small": {
        "en": "end_silence_ms is too small — phrases will be torn into pieces",
        "ru": "end_silence_ms слишком мал — фразы будут рваться на куски",
    },
    "cfg.max_utterance_big": {
        "en": "max_utterance_ms={value}: whisper's window is exactly 30 s. A longer "
              "fragment is processed in several passes — latency grows and accuracy "
              "at the seams drops.",
        "ru": "max_utterance_ms={value}: у whisper окно ровно 30 с. Более длинный "
              "фрагмент он разберёт несколькими проходами — задержка вырастет, "
              "а точность на стыках упадёт.",
    },
    "cfg.split_never": {
        "en": "split_silence_ms >= end_silence_ms: splitting on a pause will never "
              "fire, the phrase is considered finished earlier",
        "ru": "split_silence_ms >= end_silence_ms: разрез по паузе никогда не "
              "сработает, фраза раньше будет признана законченной",
    },
    "cfg.soft_over_max": {
        "en": "soft_utterance_ms > max_utterance_ms: pause-based splitting is off, "
              "long speech will be cut hard by the timer",
        "ru": "soft_utterance_ms > max_utterance_ms: нарезка по паузам отключена, "
              "длинная речь будет резаться жёстко по таймеру",
    },

    # --- сохранённые настройки --------------------------------------------
    "settings.unreadable": {
        "en": "could not read the settings file {path} ({error}) — using defaults",
        "ru": "не удалось прочитать файл настроек {path} ({error}) — беру значения "
              "по умолчанию",
    },
    "settings.bad_value": {
        "en": "settings: {field}={value!r} is not a valid value, using {fallback!r}",
        "ru": "настройки: {field}={value!r} — недопустимое значение, беру {fallback!r}",
    },
    "settings.model_gone": {
        "en": "the saved model folder {path} is gone — switching to {fallback}",
        "ru": "сохранённая папка модели {path} исчезла — беру {fallback}",
    },
    "settings.save_failed": {
        "en": "could not save settings to {path}: {error}",
        "ru": "не удалось сохранить настройки в {path}: {error}",
    },

    # --- сохранение расшифровки --------------------------------------------
    "transcript.header": {"en": "# Call {started}", "ru": "# Звонок {started}"},

    # --- окно ---------------------------------------------------------------
    "ui.title": {"en": "Call transcript", "ru": "Транскрипция звонка"},
    "ui.copy_all": {"en": "Copy all", "ru": "Скопировать всё"},
    "ui.clear": {"en": "Clear", "ru": "Очистить"},
    "ui.pause": {"en": "Pause", "ru": "Пауза"},
    "ui.resume": {"en": "Resume", "ru": "Продолжить"},
    "ui.speech_caption": {"en": "Speech:", "ru": "Речь:"},
    "ui.interface_caption": {"en": "Interface:", "ru": "Интерфейс:"},
    "ui.model_caption": {"en": "Model:", "ru": "Модель:"},
    "ui.browse_model": {"en": "Browse...", "ru": "Выбрать папку..."},
    "ui.loading_model": {"en": "Loading the model...", "ru": "Загружаю модель..."},
    "ui.loading_named": {"en": "Loading {model}...", "ru": "Загружаю {model}..."},
    "ui.listening": {"en": "Listening... [{device}]", "ru": "Слушаю... [{device}]"},
    "ui.queued": {"en": " (queued: {count})", "ru": " (в очереди: {count})"},
    "ui.paused": {
        "en": "Paused — audio is not being recorded",
        "ru": "Пауза — звук не пишется",
    },
    "ui.model_failed": {"en": "The model did not load", "ru": "Модель не загрузилась"},
    "ui.error_seen": {
        "en": "Error — see the message window",
        "ru": "Ошибка — см. окно с сообщением",
    },
    "ui.copied": {"en": "Copied to the clipboard", "ru": "Скопировано в буфер обмена"},
    "ui.nothing_to_copy": {"en": "Nothing to copy", "ru": "Нечего копировать"},
    "ui.cleared": {
        "en": "Window cleared (the file on disk is untouched)",
        "ru": "Окно очищено (файл на диске не тронут)",
    },
    "ui.stopping": {
        "en": "Stopping, finishing the queue...",
        "ru": "Останавливаюсь, дописываю очередь...",
    },
    "ui.language_set": {
        "en": "Speech language: {language}",
        "ru": "Язык распознавания: {language}",
    },
    "ui.language_auto": {
        "en": "Auto: the language is detected per phrase — unreliable on short ones",
        "ru": "Авто: язык определяется по каждой фразе — на коротких ошибается",
    },
    "ui.interface_set": {
        "en": "Interface language: {language}",
        "ru": "Язык интерфейса: {language}",
    },
    "ui.model_requested": {
        "en": "Switching to {model} — the current phrase will finish on the old one",
        "ru": "Переключаюсь на {model} — текущая фраза дойдёт на прежней",
    },
    "ui.model_dialog_title": {
        "en": "Select a faster-whisper model folder",
        "ru": "Выберите папку с моделью faster-whisper",
    },
    "ui.model_rejected": {
        "en": "Not a model folder: {reason}",
        "ru": "Это не папка с моделью: {reason}",
    },

    # --- проверка папки с моделью -------------------------------------------
    "model.not_a_dir": {
        "en": "{path} is not a folder",
        "ru": "{path} — не папка",
    },
    "model.missing_files": {
        "en": "{path} has no {files} — a faster-whisper folder is a converted "
              "CTranslate2 model, not the original .pt file",
        "ru": "в {path} нет {files} — папка faster-whisper это модель, "
              "переведённая в CTranslate2, а не исходный файл .pt",
    },
    "model.no_tokenizer": {
        "en": "{path} has no tokenizer (tokenizer.json or vocabulary.txt)",
        "ru": "в {path} нет токенизатора (tokenizer.json или vocabulary.txt)",
    },

    # --- командная строка ----------------------------------------------------
    "cli.description": {
        "en": "CallCribe — offline live transcription of a call: your microphone "
              "and the system audio at once.",
        "ru": "CallCribe — живая офлайн-расшифровка звонка: микрофон и системный "
              "звук одновременно.",
    },
    "cli.lang_help": {
        "en": "speech language at startup (default: the saved one); it can be "
              "switched in the window at any time",
        "ru": "язык распознавания при старте (по умолчанию — сохранённый); "
              "в самом окне его можно переключить в любой момент",
    },
    "cli.ui_lang_help": {
        "en": "interface language at startup (default: the saved one)",
        "ru": "язык интерфейса при старте (по умолчанию — сохранённый)",
    },
    "cli.model_help": {
        "en": "whisper model: a size name or a path to a faster-whisper folder "
              "(default: the saved one)",
        "ru": "модель whisper: имя размера или путь к папке faster-whisper "
              "(по умолчанию — сохранённая)",
    },
    "cli.windows_only": {
        "en": "CallCribe only runs on Windows: system audio is captured through "
              "WASAPI loopback.",
        "ru": "CallCribe работает только на Windows: захват системного звука "
              "сделан через WASAPI loopback.",
    },
}

# Метка канала -> ключ каталога. Отдельным словарём, а не склейкой
# "label." + key: собранный на лету ключ не найдёт ни поиск по коду, ни
# проверка каталога в selftest.
_SPEAKER_KEYS = {"me": "label.me", "them": "label.them"}


# ---------------------------------------------------------------------------
# ПЕРЕВОДЧИК
# ---------------------------------------------------------------------------


class Translator:
    """Ключ -> надпись на выбранном языке.

    Пишет сюда поток окна, читают четыре рабочих потока, поэтому язык
    закрыт замком — по той же причине, что и LanguageSetting.
    """

    def __init__(self, code: str = DEFAULT_UI_LANGUAGE):
        self._code = code if code in UI_LANGUAGE_CODES else DEFAULT_UI_LANGUAGE
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            return self._code

    def set(self, code: str) -> None:
        with self._lock:
            self._code = code if code in UI_LANGUAGE_CODES else DEFAULT_UI_LANGUAGE

    def __call__(self, key: str, **kwargs: object) -> str:
        entry = MESSAGES.get(key)
        if entry is None:
            # Ключ наружу — это заметно и чинится сразу. Молча отдавать
            # пустую строку хуже: сообщение просто исчезнет.
            return key
        template = entry.get(self.get()) or entry.get(DEFAULT_UI_LANGUAGE, key)
        if not kwargs:
            return template
        try:
            return template.format(**kwargs)
        except (KeyError, IndexError, ValueError):
            # Надпись важнее подстановки: сообщение об ошибке не должно
            # само падать в потоке, который его составляет.
            return template


_translator = Translator()


def t(key: str, **kwargs: object) -> str:
    return _translator(key, **kwargs)


def set_language(code: str) -> None:
    _translator.set(code)


def get_language() -> str:
    return _translator.get()


def speaker(label: str) -> str:
    """Метка канала -> подпись. Незнакомая метка возвращается как есть."""
    key = _SPEAKER_KEYS.get(label)
    return t(key) if key else label


def placeholders(template: str) -> frozenset[str]:
    """Имена подстановок в строке каталога — для сверки языков между собой."""
    return frozenset(
        name for _, name, _, _ in Formatter().parse(template) if name
    )
