"""Настройки, которые пользователь выбрал сам, — и они переживают перезапуск.

Config — это значения по умолчанию, вшитые в код: их правят, открывая
файл. Здесь другое: три вещи, которые меняют из окна во время работы
(язык интерфейса, язык распознавания, модель), и они обязаны сохраниться
до следующего запуска, иначе выбор модели придётся делать каждый раз.

Файл лежит в %APPDATA%\\CallCribe\\settings.json. Ни одна ошибка чтения не
должна мешать запуску: испорченный или чужой файл — это предупреждение и
значения по умолчанию, а не отказ стартовать.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .config import CFG, LANGUAGE_CODES, WHISPER_MODELS
from .i18n import UI_LANGUAGE_CODES

# Ключ каталога i18n и подстановки к нему. Переводится не здесь: язык
# интерфейса берётся из этого же файла, то есть на момент разбора ещё
# не известен.
Problem = tuple[str, dict]

MAX_CUSTOM_MODELS = 8
"""Сколько выбранных вручную папок помнить. Список — история выбора, а не
хранилище: без предела он растёт молча и выпадающий список становится
непригодным."""

# Что faster-whisper обязан найти в папке модели. Токенизатор проверяем
# отдельно: имён у него два, и без него библиотека молча полезет за ним
# на Hugging Face — в офлайн-приложении это не отказ, а зависание.
_REQUIRED_FILES = ("model.bin", "config.json")
_TOKENIZERS = ("tokenizer.json", "vocabulary.txt", "vocabulary.json")


def default_path() -> Path:
    base = os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".config"
    return root / "CallCribe" / "settings.json"


def is_local_model(value: str) -> bool:
    """Путь к папке, а не имя размера вроде large-v3."""
    return value not in WHISPER_MODELS


def model_problem(value: str) -> Problem | None:
    """Чего не хватает в папке, чтобы faster-whisper её принял.

    Имена размеров не проверяем: их библиотека скачивает сама, и судить
    об их годности отсюда нечем.
    """
    if not is_local_model(value):
        return None

    path = Path(value)
    if not path.is_dir():
        return ("model.not_a_dir", {"path": value})

    missing = [name for name in _REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        return ("model.missing_files", {"path": path.name, "files": ", ".join(missing)})
    if not any((path / name).is_file() for name in _TOKENIZERS):
        return ("model.no_tokenizer", {"path": path.name})
    return None


def model_display(value: str) -> str:
    """Короткая подпись для выпадающего списка.

    Полный путь в список ставить нельзя: он шире окна, и список
    превращается в горизонтальную простыню.
    """
    if not is_local_model(value):
        return value
    path = Path(value)
    return path.name or value


def model_labels(values: list[str]) -> dict[str, str]:
    """Подпись -> значение для выпадающего списка, без совпадающих подписей.

    Имена папок вполне повторяются: две сборки large-v3 в разных
    каталогах дадут одну и ту же подпись, и выбрать вторую станет
    невозможно — список вернёт первую. Поэтому повтор уточняется
    родительской папкой, а если совпала и она — номером.
    """
    labels: dict[str, str] = {}
    for value in values:
        base = model_display(value)
        name = base
        if name in labels:
            name = f"{base} ({Path(value).parent.name})"
        suffix = 2
        while name in labels:
            name = f"{base} ({suffix})"
            suffix += 1
        labels[name] = value
    return labels


@dataclass
class Settings:
    """Значения по умолчанию берём из Config, а не повторяем числом.

    Иначе их станет два: правишь Config, а приложение при первом запуске
    всё равно берёт своё.
    """

    ui_language: str = field(default_factory=lambda: CFG.ui_language)
    language: str | None = field(default_factory=lambda: CFG.language)
    whisper_model: str = field(default_factory=lambda: CFG.whisper_model)
    custom_models: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "ui_language": self.ui_language,
            "language": self.language,
            "whisper_model": self.whisper_model,
            "custom_models": list(self.custom_models),
        }

    def remember_model(self, value: str) -> None:
        """Поставить папку в начало истории, без повторов."""
        if not is_local_model(value):
            return
        remaining = [item for item in self.custom_models if item != value]
        self.custom_models = [value, *remaining][:MAX_CUSTOM_MODELS]


def _parse(raw: dict) -> tuple[Settings, list[Problem]]:
    """Разбор прочитанного словаря. Плохое поле — предупреждение и умолчание."""
    data = Settings()
    problems: list[Problem] = []

    def complain(field_name: str, value: object, fallback: object) -> None:
        problems.append(
            ("settings.bad_value",
             {"field": field_name, "value": value, "fallback": fallback})
        )

    if "ui_language" in raw:
        value = raw["ui_language"]
        if value in UI_LANGUAGE_CODES:
            data.ui_language = value
        else:
            complain("ui_language", value, data.ui_language)

    # Язык распознавания: None здесь — это «Авто», полноправное значение,
    # а не «не задано». Поэтому смотрим на наличие ключа, а не на истинность.
    if "language" in raw:
        value = raw["language"]
        if value in LANGUAGE_CODES:
            data.language = value
        else:
            complain("language", value, data.language)

    if "whisper_model" in raw:
        value = raw["whisper_model"]
        if isinstance(value, str) and value.strip():
            data.whisper_model = value
        else:
            complain("whisper_model", value, data.whisper_model)

    if "custom_models" in raw:
        value = raw["custom_models"]
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            data.custom_models = value[:MAX_CUSTOM_MODELS]
        else:
            complain("custom_models", value, data.custom_models)

    # Папку могли переименовать или отключить диск. Тогда выбор нужно
    # снять сейчас, а не ловить отказ загрузки через полминуты.
    if is_local_model(data.whisper_model) and not Path(data.whisper_model).is_dir():
        problems.append(
            ("settings.model_gone",
             {"path": data.whisper_model, "fallback": CFG.whisper_model})
        )
        data.whisper_model = CFG.whisper_model

    return data, problems


class SettingsStore:
    """Файл настроек: прочитать при старте, переписать при каждом выборе.

    Пишем сразу, а не при выходе: приложение живёт весь звонок и вполне
    может не дожить до штатного закрытия, а потерять из-за этого выбранную
    модель обиднее всего.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path is not None else default_path()
        self.data = Settings()

    def load(self) -> list[Problem]:
        if not self.path.is_file():
            return []                     # первый запуск — это не проблема
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            return [("settings.unreadable",
                     {"path": str(self.path), "error": f"{type(exc).__name__}: {exc}"})]
        if not isinstance(raw, dict):
            return [("settings.unreadable",
                     {"path": str(self.path), "error": f"{type(raw).__name__}"})]

        self.data, problems = _parse(raw)
        return problems

    def save(self) -> Problem | None:
        """Возвращает описание сбоя или None. Наружу не бросает: не сохранить
        настройку — досадно, но уронить из-за этого окно нельзя."""
        temp = self.path.with_name(self.path.name + ".tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp.write_text(
                json.dumps(self.data.to_json(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            # Замена целиком: оборванная запись не должна оставить
            # полуфайл, который в следующий раз не прочитается.
            os.replace(temp, self.path)
        except OSError as exc:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                pass
            return ("settings.save_failed",
                    {"path": str(self.path), "error": f"{type(exc).__name__}: {exc}"})
        return None

    def update(self, **changes: object) -> Problem | None:
        """Записать изменения и тут же сохранить."""
        for name, value in changes.items():
            setattr(self.data, name, value)
        return self.save()
