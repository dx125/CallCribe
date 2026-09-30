"""Живой список аудиоустройств Windows — чтобы заметить, что он поменялся.

Зачем отдельно от PortAudio. PortAudio читает список устройств один раз,
при инициализации, и дальше показывает его, даже если гарнитуру давно
отключили или подключили новую. Сам он об изменениях не узнаёт — а
приложение живёт весь звонок, за который Bluetooth-наушники успевают
переподключиться, звонилка — перевести звук на другой выход, а Windows —
поменять устройство по умолчанию. После этого захват либо слушает выход,
по которому звонка уже нет, либо не получает ни одного пакета: у одного
из собеседников расшифровка просто прекращается, а перезапуск «чинит».

Здесь спрашиваем саму Windows (MMDevice API) — каждый раз заново, без
кэша. Сравнивать достаточно отпечатка: набор активных устройств плюс
устройства по умолчанию. Разница — повод пересоздать захват.

COM зовётся напрямую через ctypes, без comtypes: нужно четыре метода, и
ради них тянуть зависимость незачем. Вызывать из потока с COM-апартаментом
(см. audio._ComApartment).
"""

from __future__ import annotations

import ctypes
import sys

_E_RENDER, _E_CAPTURE, _E_ALL = 0, 1, 2
_E_CONSOLE = 0
_DEVICE_STATE_ACTIVE = 0x1
_CLSCTX_ALL = 0x17

_CLSID_MM_DEVICE_ENUMERATOR = "{BCDE0395-E52F-467C-8E3D-C4579291692E}"
_IID_IMM_DEVICE_ENUMERATOR = "{A95664D2-9614-4F35-A746-DE8DB63617E6}"

# Номера методов в таблицах виртуальных функций (после трёх из IUnknown).
_RELEASE = 2
_ENUM_AUDIO_ENDPOINTS = 3      # IMMDeviceEnumerator
_GET_DEFAULT_ENDPOINT = 4      # IMMDeviceEnumerator
_COLLECTION_GET_COUNT = 3      # IMMDeviceCollection
_COLLECTION_ITEM = 4           # IMMDeviceCollection
_DEVICE_GET_ID = 5             # IMMDevice

Signature = frozenset


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> _GUID:
    value = _GUID()
    hr = ctypes.windll.ole32.CLSIDFromString(ctypes.c_wchar_p(text), ctypes.byref(value))
    if hr != 0:
        raise OSError(f"CLSIDFromString failed: 0x{hr & 0xFFFFFFFF:08X}")
    return value


def _method(obj: ctypes.c_void_p, index: int, *argtypes):
    vtable = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    prototype = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)
    return prototype(vtable[index])


def _release(obj: ctypes.c_void_p) -> None:
    if obj:
        _method(obj, _RELEASE)(obj)


def _check(hr: int, what: str) -> None:
    if hr < 0:
        raise OSError(f"{what} failed: 0x{hr & 0xFFFFFFFF:08X}")


def _device_id(device: ctypes.c_void_p) -> str:
    text = ctypes.c_wchar_p()
    hr = _method(device, _DEVICE_GET_ID, ctypes.POINTER(ctypes.c_wchar_p))(
        device, ctypes.byref(text)
    )
    _check(hr, "IMMDevice::GetId")
    try:
        return text.value or ""
    finally:
        ctypes.windll.ole32.CoTaskMemFree(text)


def _default_id(enumerator: ctypes.c_void_p, flow: int) -> str:
    device = ctypes.c_void_p()
    hr = _method(enumerator, _GET_DEFAULT_ENDPOINT, ctypes.c_int, ctypes.c_int,
                 ctypes.POINTER(ctypes.c_void_p))(
        enumerator, flow, _E_CONSOLE, ctypes.byref(device)
    )
    if hr < 0:           # устройства по умолчанию нет — это тоже состояние
        return ""
    try:
        return _device_id(device)
    finally:
        _release(device)


def endpoint_signature() -> Signature | None:
    """Отпечаток текущих аудиоустройств или None, если спросить не удалось.

    None — не «ничего нет», а «не знаем»: тогда захват полагается только
    на то, что видит сам (остановившийся поток), и лишний раз ничего не
    пересоздаёт.
    """
    if sys.platform != "win32":
        return None
    enumerator = ctypes.c_void_p()
    collection = ctypes.c_void_p()
    try:
        clsid = _guid(_CLSID_MM_DEVICE_ENUMERATOR)
        iid = _guid(_IID_IMM_DEVICE_ENUMERATOR)
        hr = ctypes.windll.ole32.CoCreateInstance(
            ctypes.byref(clsid), None, _CLSCTX_ALL, ctypes.byref(iid),
            ctypes.byref(enumerator),
        )
        _check(hr, "CoCreateInstance(MMDeviceEnumerator)")

        hr = _method(enumerator, _ENUM_AUDIO_ENDPOINTS, ctypes.c_int, ctypes.c_ulong,
                     ctypes.POINTER(ctypes.c_void_p))(
            enumerator, _E_ALL, _DEVICE_STATE_ACTIVE, ctypes.byref(collection)
        )
        _check(hr, "EnumAudioEndpoints")

        count = ctypes.c_uint()
        hr = _method(collection, _COLLECTION_GET_COUNT, ctypes.POINTER(ctypes.c_uint))(
            collection, ctypes.byref(count)
        )
        _check(hr, "IMMDeviceCollection::GetCount")

        items: set[str] = set()
        for index in range(count.value):
            device = ctypes.c_void_p()
            hr = _method(collection, _COLLECTION_ITEM, ctypes.c_uint,
                         ctypes.POINTER(ctypes.c_void_p))(
                collection, index, ctypes.byref(device)
            )
            if hr < 0:
                continue
            try:
                items.add(_device_id(device))
            finally:
                _release(device)

        items.add("default-render:" + _default_id(enumerator, _E_RENDER))
        items.add("default-capture:" + _default_id(enumerator, _E_CAPTURE))
        return frozenset(items)
    except (OSError, AttributeError, ValueError):
        return None
    finally:
        _release(collection)
        _release(enumerator)
