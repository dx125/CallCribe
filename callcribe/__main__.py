"""Точка входа: python -m callcribe"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .app import load_settings, run
from .config import CFG, COMPUTE_TYPES, LANGUAGE_CODES, WHISPER_DEVICES
from .i18n import UI_LANGUAGE_CODES, set_language, t
from .settings import model_problem
from .status import use_utf8_console

# В командной строке None не напишешь, поэтому «Авто» здесь зовётся auto.
_AUTO_ARG = "auto"
_LANG_ARGS = (_AUTO_ARG,) + tuple(code for code in LANGUAGE_CODES if code)


def main() -> None:
    use_utf8_console()

    # Настройки читаем ДО разбора аргументов: в них лежит язык интерфейса,
    # а на нём должна выйти и справка по ключам.
    store, cfg, problems = load_settings(CFG)

    parser = argparse.ArgumentParser(prog="callcribe", description=t("cli.description"))
    parser.add_argument("--lang", choices=_LANG_ARGS, help=t("cli.lang_help"))
    parser.add_argument("--ui-lang", choices=UI_LANGUAGE_CODES, help=t("cli.ui_lang_help"))
    parser.add_argument("--model", metavar="NAME|PATH", help=t("cli.model_help"))
    parser.add_argument("--device", choices=WHISPER_DEVICES, help=t("cli.device_help"))
    parser.add_argument("--compute", choices=COMPUTE_TYPES, help=t("cli.compute_help"))
    args = parser.parse_args()

    # Разбор аргументов раньше проверки платформы: --help должен работать
    # где угодно, а не только там, где приложение способно запуститься.
    if sys.platform != "win32":
        sys.exit(t("cli.windows_only"))

    # Ключи командной строки задают то же, что выпадающие списки в окне:
    # на этот запуск они видны в окне как выбранные, а в файл настроек
    # попадут, как только в окне поменяют хоть что-нибудь.
    if args.ui_lang:
        store.data.ui_language = args.ui_lang
        set_language(args.ui_lang)
    if args.lang:
        store.data.language = None if args.lang == _AUTO_ARG else args.lang
    if args.model:
        problem = model_problem(args.model)
        if problem is not None:
            # Лучше отказать сразу, чем через полминуты загрузки.
            key, params = problem
            sys.exit(t(key, **params))
        store.data.remember_model(args.model)
        store.data.whisper_model = args.model

    # Устройство и формат вычислений, в отличие от языков и модели, в файл
    # настроек не едут: это свойства машины, а не выбор человека, и меняют
    # их когда что-то пошло не так. Пусть влияют на один запуск.
    compute = cfg.whisper_compute
    if args.compute:
        compute = "" if args.compute == _AUTO_ARG else args.compute

    cfg = dataclasses.replace(
        cfg,
        ui_language=store.data.ui_language,
        language=store.data.language,
        whisper_model=store.data.whisper_model,
        whisper_device=args.device or cfg.whisper_device,
        whisper_compute=compute,
    )
    run(cfg, store, problems)


if __name__ == "__main__":
    main()
