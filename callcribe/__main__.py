"""Точка входа: python -m callcribe

Запуск двухэтажный. Сначала поднимается наблюдатель (supervisor.py) — он
лёгкий и ничего опасного не трогает, — и уже он запускает этот же модуль
вторым процессом, где работает приложение. Так падение приложения, даже
такое, какого Python не видит, заканчивается понятным окном и отчётом, а
не молча исчезнувшим окном.
"""

from __future__ import annotations

import logging
import os
import sys

from . import diagnostics

# В командной строке None не напишешь, поэтому «Авто» здесь зовётся auto.
_AUTO_ARG = "auto"


def _supervised() -> bool:
    """Нужен ли наблюдатель: не нужен в самом рабочем процессе и когда
    его отключили руками (отладка)."""
    return not (
        os.environ.get(diagnostics.CHILD_ENV) or os.environ.get(diagnostics.NO_SUPERVISOR_ENV)
    )


def fail(message: str) -> None:
    """Отказ, о котором человек обязан узнать, — и под pythonw тоже.

    sys.exit(сообщение) печатает в stderr, а у pythonw его нет: ярлык с
    негодным --model просто не открывался бы. Поэтому ещё и окно."""
    logging.getLogger("callcribe").error("%s", message)
    try:
        print(message, file=sys.stderr, flush=True)
    except (OSError, ValueError, AttributeError):
        pass
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("CallCribe", message)
        root.destroy()
    except Exception:
        pass
    sys.exit(diagnostics.EXIT_HANDLED)


def run_app() -> None:
    """Рабочий процесс: разбор ключей и само приложение."""
    import argparse
    import dataclasses

    from .status import use_utf8_console

    use_utf8_console()
    # Журнал и перехват сбоев — до всего остального: самое неприятное
    # случается как раз при старте, пока окна ещё нет.
    diagnostics.install(console=sys.stdout is not None)

    from .app import load_settings, run
    from .config import CFG, COMPUTE_TYPES, LANGUAGE_CODES, WHISPER_DEVICES
    from .i18n import UI_LANGUAGE_CODES, set_language, t
    from .settings import model_problem

    lang_args = (_AUTO_ARG,) + tuple(code for code in LANGUAGE_CODES if code)

    # Настройки читаем ДО разбора аргументов: в них лежит язык интерфейса,
    # а на нём должна выйти и справка по ключам.
    store, cfg, problems = load_settings(CFG)

    parser = argparse.ArgumentParser(prog="callcribe", description=t("cli.description"))
    parser.add_argument("--lang", choices=lang_args, help=t("cli.lang_help"))
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
            fail(t(key, **params))
        store.data.remember_model(args.model)
        store.data.whisper_model = args.model

    # Формат вычислений в файл не едет: это свойство машины, и его ключ —
    # инструмент на один запуск, когда что-то пошло не так. Устройство —
    # тоже только на этот запуск, но в окне его можно выбрать насовсем.
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
    diagnostics.shutdown_clean()


def main() -> None:
    if _supervised():
        from .supervisor import supervise

        sys.exit(supervise(sys.argv[1:]))
    run_app()


if __name__ == "__main__":
    main()
