"""Глобальная горячая клавиша: F8 работает и когда окно не в фокусе.

Нужна ровно для одного сценария: расшифровку скидывают в переписку по ходу
звонка. Щёлкнули в чат — окно CallCribe фокус потеряло, и обычная привязка
Tk до него уже не доходит. Возвращаться мышью к окну, жать F8, возвращаться
в чат — это три действия вместо одного, причём посреди разговора.

Просим у системы ОДНУ клавишу через RegisterHotKey, а не слежку за всем
вводом: перехват клавиатуры целиком решал бы ту же задачу, но требовал бы
доверия к приложению, которое читает каждое нажатие, — и объяснений с
антивирусом. RegisterHotKey не нужны ни права администратора, ни хуки.

Нажатие уезжает в очередь, а не в обработчик напрямую: Tkinter не
потокобезопасен, а в окне уже есть опрос раз в 100 мс (ui.poll), который
забирает всё накопившееся. Для копирования в буфер задержка незаметна,
зато из чужого потока к Tk не обращается никто.
"""

from __future__ import annotations

import ctypes
import queue
import sys
import threading

from .i18n import t
from .status import Notifier

VK_F8 = 0x77
"""Виртуальный код F8. Имя клавиши в документации — F8, код — 0x77."""

_WM_HOTKEY = 0x0312
_PM_REMOVE = 0x0001
_MOD_NOREPEAT = 0x4000
"""Без него зажатая клавиша сыплет сообщениями, и одно нажатие
превращается в десяток копирований."""

_HOTKEY_ID = 0xCB01          # свой id в пределах потока, лишь бы не 0
_ERROR_HOTKEY_ALREADY_REGISTERED = 1409


def _user32():
    """Загружаем по требованию: на не-Windows этого DLL просто нет."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    lib = ctypes.WinDLL("user32", use_last_error=True)
    lib.RegisterHotKey.argtypes = [
        wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT,
    ]
    lib.RegisterHotKey.restype = wintypes.BOOL
    lib.UnregisterHotKey.argtypes = [wintypes.HWND, ctypes.c_int]
    lib.UnregisterHotKey.restype = wintypes.BOOL
    lib.PeekMessageW.argtypes = [
        ctypes.POINTER(wintypes.MSG), wintypes.HWND,
        wintypes.UINT, wintypes.UINT, wintypes.UINT,
    ]
    lib.PeekMessageW.restype = wintypes.BOOL
    return lib


class HotkeyListener(threading.Thread):
    """Держит глобальную клавишу и кладёт каждое нажатие в очередь.

    Клавиша принадлежит ПОТОКУ, который её занял: RegisterHotKey с пустым
    окном адресует сообщения очереди вызвавшего потока. Поэтому и занимаем,
    и читаем, и освобождаем её здесь же — отдать регистрацию одному потоку,
    а чтение другому нельзя.
    """

    def __init__(
        self,
        presses: "queue.Queue[float]",
        stop_event: threading.Event,
        notifier: Notifier,
        key: int = VK_F8,
        key_name: str = "F8",
    ):
        super().__init__(daemon=True, name="hotkey")
        self.presses = presses
        self.stop_event = stop_event
        self.notifier = notifier
        self.key = key
        self.key_name = key_name
        # Взводится, когда клавиша занята нами. Окно по нему знает, стоит ли
        # обещать пользователю работу вне фокуса.
        self.active = threading.Event()
        # Взводится всегда — и при удаче, и при отказе: тот, кто ждёт
        # результата регистрации, не должен ждать его до таймаута впустую.
        self.settled = threading.Event()

    def run(self) -> None:
        lib = _user32()
        if lib is None:
            self.settled.set()
            return

        from ctypes import wintypes

        message = wintypes.MSG()
        # Первый вызов PeekMessage заводит потоку очередь сообщений — ту
        # самую, в которую RegisterHotKey будет складывать нажатия.
        lib.PeekMessageW(ctypes.byref(message), None, 0, 0, _PM_REMOVE)

        if not lib.RegisterHotKey(None, _HOTKEY_ID, _MOD_NOREPEAT, self.key):
            code = ctypes.get_last_error()
            # Клавишу уже кто-то занял — это не сбой приложения, а занятая
            # клавиша. Остальное живёт как жило, только F8 работает лишь
            # когда окно в фокусе.
            if code == _ERROR_HOTKEY_ALREADY_REGISTERED:
                self.notifier.warn(t("hotkey.taken", hotkey=self.key_name))
            else:
                self.notifier.warn(
                    t("hotkey.failed", hotkey=self.key_name, error=code)
                )
            self.settled.set()
            return

        self.active.set()
        self.settled.set()
        try:
            # Ждём остановки, попутно вычерпывая нажатия. Блокирующий
            # GetMessage читался бы красивее, но разбудить его можно только
            # сообщением в тот же поток, то есть ещё одним вызовом WinAPI;
            # опрос раз в 50 мс стоит пренебрежимо мало и останавливается
            # тем же stop_event, что и всё остальное в приложении.
            while not self.stop_event.wait(0.05):
                while lib.PeekMessageW(
                    ctypes.byref(message), None, _WM_HOTKEY, _WM_HOTKEY, _PM_REMOVE
                ):
                    if message.wParam == _HOTKEY_ID:
                        self.presses.put(0.0)
        finally:
            self.active.clear()
            lib.UnregisterHotKey(None, _HOTKEY_ID)
