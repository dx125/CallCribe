"""Журнал, ловля сбоев и отчёт о падении.

Приложение запускают ярлыком через pythonw, а у pythonw нет ни консоли, ни
stderr. Всё, что Python печатает при сбое, там уходит в никуда, и снаружи
падение выглядит как «окно просто закрылось, ошибок нет». Поэтому:

  * журнал пишется в файл всегда — и сообщения, которые видел
    пользователь, и трассировки, и то, что напечатал бы print;
  * необработанное исключение в любом потоке попадает в журнал целиком;
  * от ошибок уровня процесса (access violation внутри CTranslate2 или
    драйвера — мимо любого except) Python спасти не может, но
    faulthandler успевает записать стеки всех потоков в отдельный файл;
  * текущая фаза работы («гружу модель на cuda/float16», «слушаю»)
    лежит в session.json. Процесс-наблюдатель (supervisor.py) по коду
    выхода понимает, что процесс упал, и по фазе — что тот в этот момент
    делал; отсюда и понятное человеку сообщение, и предложение вроде
    «запустить на процессоре».

Всё это лежит в %LOCALAPPDATA%\\CallCribe: журнал — не настройки, и в
перемещаемый профиль (APPDATA\\Roaming) ему незачем.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import logging.handlers
import os
import platform
import sys
import threading
import time
from pathlib import Path

_LOG = logging.getLogger("callcribe")

CHILD_ENV = "CALLCRIBE_CHILD"
"""Выставлен наблюдателем для рабочего процесса. См. supervisor.py."""

NO_SUPERVISOR_ENV = "CALLCRIBE_NO_SUPERVISOR"
"""Запуск без наблюдателя — для отладки, когда второй процесс мешает."""

EXIT_HANDLED = 10
"""Ошибка уже показана человеку (окном), наблюдателю добавить нечего.

Не 1 и не 3: единицу Python возвращает на любое необработанное исключение,
а тройку — abort() из рантайма MSVC, то есть как раз настоящий сбой."""

EXIT_UNHANDLED = 11
"""Необработанное исключение Python. Трассировка уже в журнале."""

MAX_REPORTS = 20
"""Сколько отчётов о падениях хранить. Старые удаляются: отчёт нужен про
свежий сбой, а не архив за год."""


# ---------------------------------------------------------------------------
# ПУТИ
# ---------------------------------------------------------------------------


def app_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    root = Path(base) if base else Path.home() / ".callcribe"
    return root / "CallCribe"


def logs_dir() -> Path:
    return app_dir() / "logs"


def log_path() -> Path:
    return logs_dir() / "callcribe.log"


def session_path() -> Path:
    return app_dir() / "session.json"


# ---------------------------------------------------------------------------
# ФАЗА РАБОТЫ
# ---------------------------------------------------------------------------

_session: dict | None = None
_session_lock = threading.Lock()


def _write_session() -> None:
    """Атомарно: наблюдатель может читать файл в любой момент, в том
    числе в тот, когда процесс умер посреди записи."""
    if _session is None:
        return
    path = session_path()
    temp = path.with_name(path.name + ".tmp")
    try:
        temp.write_text(json.dumps(_session, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        pass


def set_phase(phase: str, **details: object) -> None:
    """Что процесс делает сейчас. Без инициализации ничего не делает —
    так модулям не нужно знать, запущены ли они из приложения или из
    selftest."""
    with _session_lock:
        if _session is None:
            return
        _session["phase"] = phase
        _session["details"] = {k: v for k, v in details.items() if v is not None}
        _session["phase_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        _write_session()
    _LOG.info("phase: %s %s", phase, details or "")


def remember(**values: object) -> None:
    """Дописать в сессию то, что понадобится в отчёте (путь расшифровки)."""
    with _session_lock:
        if _session is None:
            return
        _session.update(values)
        _write_session()


def read_session() -> dict | None:
    try:
        raw = json.loads(session_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return raw if isinstance(raw, dict) else None


def clear_session() -> None:
    try:
        session_path().unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# ЖУРНАЛ И ПЕРЕХВАТ СБОЕВ (рабочий процесс)
# ---------------------------------------------------------------------------


class LogStream:
    """Замена sys.stdout/stderr под pythonw: всё напечатанное — в журнал.

    Там этих потоков нет вовсе (None), и первая же печать в них роняла бы
    процесс, а трассировка необработанного исключения просто терялась."""

    is_log_stream = True

    def __init__(self, level: int):
        self._level = level
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                _LOG.log(self._level, line.rstrip())
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            _LOG.log(self._level, self._buffer.rstrip())
        self._buffer = ""

    def isatty(self) -> bool:
        return False


_fault_file = None
_notify_fatal = None


def set_fatal_sink(callback) -> None:
    """Куда ещё сообщать о сбое потока — обычно notifier.fatal, чтобы
    человек увидел его в окне, а не только в файле."""
    global _notify_fatal
    _notify_fatal = callback


def _memory_gb() -> str:
    if sys.platform != "win32":
        return "?"
    import ctypes

    class _MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemStatus()
    status.dwLength = ctypes.sizeof(_MemStatus)
    try:
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            total = status.ullTotalPhys / 2**30
            free = status.ullAvailPhys / 2**30
            return f"{total:.1f} GB total, {free:.1f} GB free"
    except Exception:
        pass
    return "?"


def gpu_summary(timeout: float = 4.0) -> str:
    """Имя видеокарты, память и драйвер — по nvidia-smi, если он есть.

    Без этой строки отчёт о падении на видеокарте почти бесполезен: сбой
    на Pascal и на RTX 40xx — это разные причины и разные лекарства."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return "no NVIDIA driver (nvidia-smi not found)"
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)   # без мигающей консоли
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=timeout, creationflags=flags,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"nvidia-smi failed: {type(exc).__name__}"
    text = (out.stdout or out.stderr or "").strip()
    return text.replace("\n", " | ") or f"nvidia-smi exit {out.returncode}"


def environment_summary() -> list[str]:
    from . import __version__

    return [
        f"CallCribe {__version__}",
        f"Python {sys.version.split()[0]} ({sys.executable})",
        f"OS {platform.platform()}",
        f"CPU {platform.processor() or '?'} ({os.cpu_count()} logical)",
        f"RAM {_memory_gb()}",
    ]


def _log_gpu_in_background() -> None:
    def work() -> None:
        _LOG.info("GPU %s", gpu_summary())

    threading.Thread(target=work, name="gpu-info", daemon=True).start()


def install(*, console: bool) -> None:
    """Включить журнал и перехват сбоев. Вызывается рабочим процессом один
    раз, до всего остального.

    console=False — нет живой консоли (pythonw): тогда stdout и stderr
    уходят в журнал, иначе печать в них пропадала бы молча.
    """
    global _session, _fault_file

    logs_dir().mkdir(parents=True, exist_ok=True)
    _prune_stale_faults()

    handler = logging.handlers.RotatingFileHandler(
        log_path(), maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s [%(threadName)s] %(message)s"
    ))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    if not console or sys.stdout is None:
        sys.stdout = LogStream(logging.INFO)
    if not console or sys.stderr is None:
        sys.stderr = LogStream(logging.ERROR)

    _LOG.info("=" * 60)
    for line in environment_summary():
        _LOG.info(line)
    _log_gpu_in_background()

    # Файл для faulthandler — свой на процесс и открытый всё время жизни:
    # в момент access violation открывать файлы уже поздно.
    fault = logs_dir() / f"fault-{os.getpid()}.log"
    try:
        _fault_file = open(fault, "w", encoding="utf-8")
        faulthandler.enable(file=_fault_file, all_threads=True)
    except OSError:
        _fault_file = None

    sys.excepthook = _excepthook
    threading.excepthook = _thread_excepthook

    # Без этого на падение Windows показывает своё окно «python.exe
    # перестал работать» — о Python, которого человек не запускал, и без
    # единого слова о том, что делать. Своё окно покажет наблюдатель.
    if sys.platform == "win32":
        import ctypes

        _SEM_FAILCRITICALERRORS = 0x0001
        _SEM_NOGPFAULTERRORBOX = 0x0002
        try:
            ctypes.windll.kernel32.SetErrorMode(
                _SEM_FAILCRITICALERRORS | _SEM_NOGPFAULTERRORBOX
            )
        except (AttributeError, OSError):
            pass

    with _session_lock:
        _session = {
            "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": "startup",
            "details": {},
            "log": str(log_path()),
            "fault": str(fault) if _fault_file else None,
        }
        _write_session()


def _excepthook(exc_type, exc, tb) -> None:
    """Необработанное исключение главного потока: в журнал и выйти кодом,
    по которому наблюдатель поймёт, что пора показать отчёт."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    _LOG.critical("unhandled exception", exc_info=(exc_type, exc, tb))
    for handler in logging.getLogger().handlers:
        handler.flush()
    os._exit(EXIT_UNHANDLED)


def _thread_excepthook(args) -> None:
    """Поток умер от исключения. Сам процесс жив, но часть работы встала
    — и об этом должен узнать не только файл, но и человек."""
    if args.exc_type is SystemExit:
        return
    name = args.thread.name if args.thread is not None else "?"
    _LOG.critical("thread %s crashed", name,
                  exc_info=(args.exc_type, args.exc_value, args.exc_traceback))
    if _notify_fatal is not None:
        try:
            _notify_fatal(f"{name}: {args.exc_type.__name__}: {args.exc_value}")
        except Exception:
            pass


def log_exception(context: str, exc: BaseException) -> None:
    """Ошибка, которую поймали и пережили, но трассировку которой терять
    нельзя: на экран идёт одна строка, в журнал — всё."""
    _LOG.error("%s", context, exc_info=(type(exc), exc, exc.__traceback__))


def shutdown_clean() -> None:
    """Приложение отработало и всё сохранило.

    Сессию НЕ удаляем, а помечаем: впереди ещё разбор интерпретатора, и
    CTranslate2 умеет упасть именно там (0xC0000409 при выгрузке). Такое
    падение уже ничего не стоит пользователю — расшифровка на диске, — и
    окном о нём пугать не нужно; но записать его стоит. Отличить его
    наблюдатель может только по этой пометке. Файл сбоя по той же причине
    остаётся открытым до самого конца: убирает его наблюдатель.
    """
    set_phase("exited")
    for handler in logging.getLogger().handlers:
        handler.flush()


def cleanup_after(session: dict | None) -> None:
    """Прибрать за процессом, который закрылся штатно (вызывает наблюдатель)."""
    fault = (session or {}).get("fault")
    if fault:
        try:
            path = Path(fault)
            if path.is_file() and path.stat().st_size == 0:
                path.unlink()
        except OSError:
            pass
    clear_session()


def _prune_stale_faults() -> None:
    """Пустые файлы сбоя от запусков без наблюдателя (или от наблюдателя,
    которого закрыли вместе с консолью)."""
    for path in logs_dir().glob("fault-*.log"):
        try:
            if path.stat().st_size == 0 and time.time() - path.stat().st_mtime > 3600:
                path.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# ОТЧЁТ О ПАДЕНИИ (наблюдатель)
# ---------------------------------------------------------------------------

_EXIT_MEANINGS = {
    0xC0000005: "access violation",
    0xC0000409: "fail-fast / stack buffer overrun",
    0xC00000FD: "stack overflow",
    0xC0000017: "out of memory",
    0xC000009A: "insufficient system resources",
    0xC000001D: "illegal instruction (CPU lacks an instruction set, e.g. AVX)",
    0xC0000135: "a required DLL was not found",
    0xC0000142: "a DLL failed to initialize",
    0xE06D7363: "unhandled C++ exception",
    0x40000015: "abort",
    3: "abort() in the C runtime",
    EXIT_UNHANDLED: "unhandled Python exception",
}


def describe_exit(code: int) -> str:
    code &= 0xFFFFFFFF
    meaning = _EXIT_MEANINGS.get(code, "unknown")
    return f"{code} (0x{code:08X}, {meaning})"


def _tail(path: Path, lines: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "(unavailable)"
    return "\n".join(text.splitlines()[-lines:])


def write_crash_report(exit_code: int, session: dict | None, extra: list[str] = ()) -> Path:
    """Собрать всё, что известно о падении, в один файл, который можно
    отправить разработчику целиком."""
    logs_dir().mkdir(parents=True, exist_ok=True)
    # Два падения за одну секунду — не экзотика (упал, перезапустили на
    # CPU, упал снова), и одинаковое имя стёрло бы первый отчёт вторым.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    report = logs_dir() / f"crash-{stamp}.txt"
    suffix = 2
    while report.exists():
        report = logs_dir() / f"crash-{stamp}-{suffix}.txt"
        suffix += 1

    parts = [
        "CallCribe crash report",
        f"time:      {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"exit code: {describe_exit(exit_code)}",
        *environment_summary(),
        *extra,
        "",
        "--- what it was doing ---",
        json.dumps(session, ensure_ascii=False, indent=2) if session else "(no session record)",
    ]

    fault = session.get("fault") if session else None
    if fault:
        parts += ["", "--- native fault (faulthandler) ---", _tail(Path(fault), 400)]

    settings = Path(os.environ.get("APPDATA", "")) / "CallCribe" / "settings.json"
    if settings.is_file():
        parts += ["", "--- settings.json ---", _tail(settings, 60)]

    parts += ["", "--- last log lines ---", _tail(log_path(), 300)]
    report.write_text("\n".join(parts) + "\n", encoding="utf-8")

    if fault:
        try:
            Path(fault).unlink(missing_ok=True)   # содержимое уже в отчёте
        except OSError:
            pass
    _prune_reports()
    return report


def _prune_reports() -> None:
    reports = sorted(logs_dir().glob("crash-*.txt"))
    for old in reports[:-MAX_REPORTS]:
        try:
            old.unlink()
        except OSError:
            pass
