"""Процесс-наблюдатель: запускает приложение и отвечает за его падение.

Зачем второй процесс. Самые тяжёлые сбои здесь — не исключения Python, а
падения уровня процесса: access violation внутри CTranslate2 (float16 на
видеокарте, которая его не умеет; кончилась видеопамять), отказ драйвера
звука. Их не ловит ни один except: процесс исчезает, и под pythonw, которым
запускает ярлык, от него не остаётся ни окна, ни строчки. Именно так
«приложение закрывается само, без ошибок».

Изнутри такое не показать — показывать уже некому. А снаружи видно всё:
код выхода говорит, ЧТО случилось, session.json — что процесс в этот момент
ДЕЛАЛ (см. diagnostics.set_phase). Этого хватает и на понятное человеку
сообщение, и на единственный полезный совет: упал на видеокарте —
запустить на процессоре.

Сам наблюдатель нарочно лёгкий: он не трогает ни звук, ни CTranslate2, ни
модель — то есть ровно то, что способно уронить процесс.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from . import diagnostics
from .i18n import set_language, t

# Коды, после которых говорить не о чем: штатный выход, ошибка разбора
# ключей (argparse уже всё напечатал), уже показанная пользователю ошибка
# и Ctrl+C в консоли.
_QUIET_EXITS = frozenset({0, 2, diagnostics.EXIT_HANDLED, 0xC000013A})

# Фазы, в которых процесс занимался моделью. Упасть в них на видеокарте —
# самый частый сбой приложения, и лекарство у него известно.
_MODEL_PHASES = frozenset({"model_load", "model_switch"})


def is_crash(code: int) -> bool:
    return (code & 0xFFFFFFFF) not in _QUIET_EXITS


def crash_kind(session: dict | None) -> tuple[str, bool]:
    """Ключ сообщения для окна и можно ли предложить запуск на CPU."""
    phase = (session or {}).get("phase") or "startup"
    details = (session or {}).get("details") or {}
    on_gpu = details.get("device") == "cuda"

    if phase in _MODEL_PHASES:
        return ("crash.gpu_load" if on_gpu else "crash.cpu_load"), on_gpu
    if phase == "listening":
        return ("crash.gpu_running" if on_gpu else "crash.running"), on_gpu
    if phase == "shutdown":
        return "crash.shutdown", False
    if phase == "audio":
        return "crash.audio", False
    return "crash.startup", False


def _child_command(argv: list[str]) -> list[str]:
    return [sys.executable, "-m", "callcribe", *argv]


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    env[diagnostics.CHILD_ENV] = "1"
    # Пакет должен найтись, из какой бы папки ни запустили: ярлык ставит
    # рабочую папку на проект, но человек может звать и откуда угодно.
    package_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (package_root, env.get("PYTHONPATH", "")) if p
    )
    return env


def _prefer_cpu() -> None:
    """Запомнить выбор «процессор» — тем же файлом, что и окно."""
    from .settings import SettingsStore

    store = SettingsStore()
    store.load()
    store.update(whisper_device="cpu")


def supervise(argv: list[str]) -> int:
    """Запускать рабочий процесс, пока он не закроется штатно или человек
    не откажется перезапускать. Возвращает код выхода для ОС."""
    # Язык окна о падении — тот же, что у приложения.
    try:
        from .settings import SettingsStore

        store = SettingsStore()
        store.load()
        set_language(store.data.ui_language)
    except Exception:
        pass

    env = _child_env()
    while True:
        diagnostics.clear_session()
        try:
            process = subprocess.Popen(_child_command(argv), env=env)
        except OSError as exc:
            _show_simple_error(t("crash.spawn_failed", error=f"{type(exc).__name__}: {exc}"))
            return 1

        try:
            code = process.wait()
        except KeyboardInterrupt:
            # Ctrl+C достался обоим процессам. Дать рабочему закончить
            # (он дописывает расшифровку) и выйти тихо.
            try:
                code = process.wait(timeout=30)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                process.kill()
                code = 0xC000013A
            return code

        session = diagnostics.read_session()
        if not is_crash(code):
            diagnostics.cleanup_after(session)
            return code

        try:
            report = diagnostics.write_crash_report(
                code, session, extra=[f"GPU {diagnostics.gpu_summary()}"]
            )
        except OSError:
            report = None
        diagnostics.clear_session()

        # Упал уже после того, как всё сохранил и закрыл окно, — при
        # выгрузке интерпретатора. Отчёт записан, а окно о «падении» после
        # того, как человек сам закрыл приложение, только напугало бы.
        if (session or {}).get("phase") == "exited":
            return code

        choice = show_crash_dialog(code, session, report)
        if choice == "cpu":
            try:
                _prefer_cpu()
            except Exception:
                pass
            continue
        if choice == "restart":
            continue
        return code


# ---------------------------------------------------------------------------
# ОКНО О ПАДЕНИИ
# ---------------------------------------------------------------------------


def crash_message(code: int, session: dict | None, report: Path | None) -> tuple[str, bool]:
    """Текст окна и можно ли предложить процессор. Отдельно от Tk, чтобы
    проверять текст без окна."""
    key, offer_cpu = crash_kind(session)
    parts = [t(key)]
    transcript = (session or {}).get("transcript")
    if transcript:
        parts.append(t("crash.transcript", path=transcript))
    if report is not None:
        parts.append(t("crash.report", path=report))
    parts.append(t("crash.code", code=diagnostics.describe_exit(code)))
    return "\n\n".join(parts), offer_cpu


def show_crash_dialog(code: int, session: dict | None, report: Path | None) -> str:
    """Возвращает "cpu", "restart" или "close"."""
    text, offer_cpu = crash_message(code, session, report)
    try:
        return _tk_dialog(text, offer_cpu, report)
    except Exception:
        # Без Tk — хотя бы системное окно: молча закрыться нельзя ни в
        # каком случае, ради этого наблюдатель и существует.
        _show_simple_error(text)
        return "close"


def _tk_dialog(text: str, offer_cpu: bool, report: Path | None) -> str:
    import tkinter as tk

    result = {"choice": "close"}
    root = tk.Tk()
    root.title(t("crash.title"))
    root.resizable(False, False)
    root.attributes("-topmost", True)

    frame = tk.Frame(root, padx=16, pady=14)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text=t("crash.title"), font=("Segoe UI", 12, "bold"),
             anchor="w").pack(fill="x")
    # Text, а не Label: путь к отчёту должно быть можно выделить и
    # скопировать — его просят прислать.
    body = tk.Text(frame, width=72, height=12, wrap="word", relief="flat",
                   font=("Segoe UI", 10), background=root.cget("background"))
    body.insert("1.0", text)
    body.configure(state="disabled")
    body.pack(fill="both", expand=True, pady=(8, 12))

    buttons = tk.Frame(frame)
    buttons.pack(fill="x")

    def choose(value: str) -> None:
        result["choice"] = value
        root.destroy()

    def open_report() -> None:
        if report is not None:
            try:
                os.startfile(report)          # noqa: S606 — открыть в Блокноте
            except OSError:
                pass

    tk.Button(buttons, text=t("crash.close"), width=12,
              command=lambda: choose("close")).pack(side="right")
    if offer_cpu:
        tk.Button(buttons, text=t("crash.use_cpu"), width=22, default="active",
                  command=lambda: choose("cpu")).pack(side="right", padx=(0, 8))
    else:
        tk.Button(buttons, text=t("crash.restart"), width=16,
                  command=lambda: choose("restart")).pack(side="right", padx=(0, 8))
    if report is not None:
        tk.Button(buttons, text=t("crash.open_report"), width=16,
                  command=open_report).pack(side="left")

    root.protocol("WM_DELETE_WINDOW", lambda: choose("close"))
    root.mainloop()
    return result["choice"]


def _show_simple_error(text: str) -> None:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "CallCribe", 0x10)
    else:
        print(text, file=sys.stderr)
