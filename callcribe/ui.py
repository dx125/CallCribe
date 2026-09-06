"""Окно с живой расшифровкой."""

from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .asr import TranscriberWorker
from .config import (
    WHISPER_MODELS,
    Config,
    LanguageSetting,
    ModelSetting,
    language_code,
    language_name,
    language_names,
)
from .i18n import (
    UI_LANGUAGES,
    set_language,
    speaker,
    t,
    ui_language_code,
    ui_language_name,
)
from .models import Line, Utterance
from .settings import SettingsStore, model_display, model_labels, model_problem
from .status import FATAL, INFO, WARN, Notice, Notifier

_POLL_MS = 100
_MAX_DRAIN = 50          # строк за один тик, чтобы не подвесить GUI
_TRANSIENT_MS = 4000     # сколько держать временное сообщение в статусе


class TranscriptWindow:
    """Живой вывод расшифровки. Текст выделяется и копируется как обычный,
    плюс кнопка «Скопировать всё»."""

    def __init__(
        self,
        cfg: Config,
        gui_queue: "queue.Queue[Line]",
        transcribe_queue: "queue.Queue[Utterance]",
        notifier: Notifier,
        stop_event: threading.Event,
        pause_event: threading.Event,
        worker: TranscriberWorker,
        sources: dict[str, str],
        language: LanguageSetting | None = None,
        model: ModelSetting | None = None,
        store: SettingsStore | None = None,
    ):
        self.cfg = cfg
        self.gui_queue = gui_queue
        self.transcribe_queue = transcribe_queue
        self.notifier = notifier
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.worker = worker
        self.sources = sources
        # Обычно это те же объекты, что отданы потоку распознавания, —
        # через них окно на него и влияет.
        self.language = language if language is not None else worker.language
        self.model = model if model is not None else worker.model_setting
        # None — настройки не сохраняем (так гоняется selftest).
        self.store = store

        self._startup_status = ""
        self._transient_until = 0.0
        self._ready_shown = False
        self._closing = False
        # На какую модель ждём переключения. Сохранять выбор нужно только
        # после того, как модель РЕАЛЬНО поднялась: иначе в файл настроек
        # уедет то, что не грузится, и следующий запуск начнётся с отказа.
        self._awaiting_model: str | None = None

        self.root = tk.Tk()
        self.root.geometry(cfg.window_geometry)

        # --- строка 1: статус и кнопки ---------------------------------
        toolbar = tk.Frame(self.root)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))

        # Управление пакуется ПЕРВЫМ, статус — последним и с expand=True.
        # pack раздаёт место в порядке упаковки: если первой идёт надпись
        # статуса, она забирает всю свою ширину, а в узком окне обрезаются
        # кнопки. Наоборот — обрезается текст статуса, что не мешает.
        self.copy_button = tk.Button(toolbar, command=self.copy_all)
        self.copy_button.pack(side="right")
        self.clear_button = tk.Button(toolbar, command=self.clear_text)
        self.clear_button.pack(side="right", padx=(0, 8))
        self.pause_button = tk.Button(toolbar, width=10, command=self.toggle_pause)
        self.pause_button.pack(side="right", padx=(0, 8))

        self.status_var = tk.StringVar()
        tk.Label(toolbar, textvariable=self.status_var, anchor="w").pack(
            side="left", fill="x", expand=True
        )

        # --- строка 2: выбор языков и модели ---------------------------
        # Отдельной строкой, а не в одной с кнопками: три подписи и три
        # списка рядом с тремя кнопками не помещаются в окно по умолчанию,
        # и первым обрезается то, что оказалось справа.
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=8, pady=(6, 0))

        self.speech_caption = tk.Label(controls)
        self.speech_caption.pack(side="left")
        self.language_var = tk.StringVar()
        self.language_picker = ttk.Combobox(
            controls,
            textvariable=self.language_var,
            state="readonly",   # только выбор из списка, руками не вписать
            width=9,
        )
        self.language_picker.pack(side="left", padx=(4, 12))
        self.language_picker.bind("<<ComboboxSelected>>", self._on_language)

        self.interface_caption = tk.Label(controls)
        self.interface_caption.pack(side="left")
        self.ui_language_var = tk.StringVar()
        self.ui_language_picker = ttk.Combobox(
            controls,
            textvariable=self.ui_language_var,
            values=[name for name, _ in UI_LANGUAGES],
            state="readonly",
            width=9,
        )
        self.ui_language_picker.pack(side="left", padx=(4, 12))
        self.ui_language_picker.bind("<<ComboboxSelected>>", self._on_ui_language)

        self.model_caption = tk.Label(controls)
        self.model_caption.pack(side="left")
        self.model_var = tk.StringVar()
        self.model_picker = ttk.Combobox(
            controls,
            textvariable=self.model_var,
            state="readonly",
            width=22,
        )
        self.model_picker.pack(side="left", padx=(4, 0))
        self.model_picker.bind("<<ComboboxSelected>>", self._on_model)
        self._model_values: dict[str, str] = {}

        # --- строка 3: откуда берётся звук ------------------------------
        self.sources_label = tk.Label(
            self.root, anchor="w", fg="#666666", font=("Segoe UI", 8)
        )
        self.sources_label.pack(fill="x", padx=10, pady=(4, 0))

        self.text = scrolledtext.ScrolledText(self.root, wrap="word", font=cfg.font)
        self.text.pack(fill="both", expand=True, padx=8, pady=8)
        self.text.configure(state="disabled")   # выделять и копировать можно, править — нет
        self.text.bind("<Control-a>", self._select_all)
        self.text.bind("<Control-A>", self._select_all)

        self._apply_texts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(_POLL_MS, self.poll)

    # ------------------------------------------------------------------
    # надписи
    # ------------------------------------------------------------------

    def _apply_texts(self) -> None:
        """Перерисовать всё, что зависит от языка интерфейса.

        Вызывается и при сборке окна, и при смене языка на ходу: второй
        раз — ровно та же работа, поэтому отдельного пути для «переключили
        язык» нет и рассинхронизироваться нечему.
        """
        self.root.title(t("ui.title"))
        self.copy_button.configure(text=t("ui.copy_all"))
        self.clear_button.configure(text=t("ui.clear"))
        self.pause_button.configure(
            text=t("ui.resume") if self.pause_event.is_set() else t("ui.pause")
        )
        self.speech_caption.configure(text=t("ui.speech_caption"))
        self.interface_caption.configure(text=t("ui.interface_caption"))
        self.model_caption.configure(text=t("ui.model_caption"))

        # Подпись «Авто» переводится, остальные языки названы на себе —
        # поэтому список пересобирается целиком, а не правится точечно.
        self.language_picker.configure(values=language_names())
        self.language_var.set(language_name(self.language.get()))
        self.ui_language_var.set(ui_language_name(self._ui_language()))

        self._rebuild_model_choices()
        self.sources_label.configure(
            text=" · ".join(
                f"{speaker(label)}: {name}" for label, name in self.sources.items()
            )
        )
        self._transient_until = 0.0
        self._refresh_status()

    def _ui_language(self) -> str:
        return self.store.data.ui_language if self.store else self.cfg.ui_language

    # ------------------------------------------------------------------
    # список моделей
    # ------------------------------------------------------------------

    def _known_models(self) -> list[str]:
        """Готовые имена, выбранные ранее папки и то, что стоит сейчас."""
        values = list(WHISPER_MODELS)
        custom = self.store.data.custom_models if self.store else []
        for value in [*custom, self.model.get()]:
            if value not in values:
                values.append(value)
        return values

    def _rebuild_model_choices(self) -> None:
        self._model_values = model_labels(self._known_models())
        self.model_picker.configure(
            values=[*self._model_values, t("ui.browse_model")]
        )
        self._sync_model_var()

    def _sync_model_var(self) -> None:
        """Показать в списке то, что реально выбрано."""
        current = self._awaiting_model or self.model.get()
        for name, value in self._model_values.items():
            if value == current:
                self.model_var.set(name)
                return
        self.model_var.set(model_display(current))

    # ------------------------------------------------------------------
    # сохранение настроек
    # ------------------------------------------------------------------

    def _save(self, **changes: object) -> None:
        if self.store is None:
            return
        problem = self.store.update(**changes)
        if problem is not None:
            key, params = problem
            self._flash(t(key, **params), 8000)

    # ------------------------------------------------------------------
    # статус
    # ------------------------------------------------------------------

    def _flash(self, text: str, duration_ms: int = _TRANSIENT_MS) -> None:
        self.status_var.set(text)
        self._transient_until = time.monotonic() + duration_ms / 1000

    def _refresh_status(self) -> None:
        """Статус собирается из состояния, а не запоминается строкой.

        Так его можно перерисовать в любой момент — например, когда
        переключили язык интерфейса, — не гадая, что там было раньше.
        """
        if self._closing:
            return
        if time.monotonic() < self._transient_until:
            return

        if self.pause_event.is_set():
            self.status_var.set(t("ui.paused"))
            return

        if self.worker.loading.is_set():
            name = self.worker.loading_name
            self.status_var.set(
                t("ui.loading_named", model=model_display(name)) if name
                else t("ui.loading_model")
            )
            return

        if not self._ready_shown:
            self.status_var.set(self._startup_status or t("ui.loading_model"))
            return

        if self.worker.failed.is_set():
            self.status_var.set(t("ui.model_failed"))
            return

        pending = self.transcribe_queue.qsize()
        suffix = t("ui.queued", count=pending) if pending else ""
        self.status_var.set(t("ui.listening", device=self.worker.device_label) + suffix)

    # ------------------------------------------------------------------
    # опрос очередей
    # ------------------------------------------------------------------

    def poll(self) -> None:
        self._drain_notices()

        if not self._ready_shown and self.worker.ready.is_set():
            self._ready_shown = True

        self._check_model_switch()
        self._drain_lines()
        self._refresh_status()
        self.root.after(_POLL_MS, self.poll)

    def _check_model_switch(self) -> None:
        """Догрузилась ли модель, которую попросили.

        Сохраняем выбор только здесь: пока модель не поднялась, записывать
        её в настройки нельзя — иначе неудачный выбор переживёт перезапуск
        и приложение будет открываться отказом.
        """
        if self._awaiting_model is None:
            return
        if self.model.pending() is not None or self.worker.loading.is_set():
            return                          # заявка ещё в пути

        loaded = self.model.get()
        if loaded == self._awaiting_model:
            self._save(whisper_model=loaded)
        # Иначе загрузка не удалась и поток вернул прежнюю модель.
        # Предупреждение про это уже показал notifier, окну остаётся
        # вернуть список к тому, что есть на самом деле.
        self._awaiting_model = None
        self._sync_model_var()

    def _drain_notices(self) -> None:
        while True:
            try:
                notice: Notice = self.notifier.queue.get_nowait()
            except queue.Empty:
                break

            if notice.level == FATAL:
                self._flash(t("ui.error_seen"), 8000)
                messagebox.showerror("CallCribe", notice.text, parent=self.root)
            elif notice.level == WARN:
                self._flash(f"⚠ {notice.text}", 6000)
            elif notice.level == INFO and not self._ready_shown:
                self._startup_status = notice.text
                self._refresh_status()

    def _drain_lines(self) -> None:
        pinned = self._at_bottom()
        drained = 0
        while drained < _MAX_DRAIN:
            try:
                line = self.gui_queue.get_nowait()
            except queue.Empty:
                break

            stamp = time.strftime("%H:%M:%S", time.localtime(line.ts))
            label = f"{speaker(line.label)}: " if self.cfg.show_speaker_labels else ""
            self._append(f"[{stamp}] {label}{line.text}\n\n")
            drained += 1

        # Доскроллить только если пользователь и так был внизу: иначе
        # вид дёргается ровно в тот момент, когда он что-то выделяет.
        if drained and pinned:
            self.text.see("end")

    # ------------------------------------------------------------------
    # действия
    # ------------------------------------------------------------------

    def _at_bottom(self) -> bool:
        try:
            return self.text.yview()[1] >= 0.999
        except tk.TclError:
            return True

    def _append(self, chunk: str) -> None:
        self.text.configure(state="normal")
        self.text.insert("end", chunk)
        self.text.configure(state="disabled")

    def _select_all(self, _event=None) -> str:
        self.text.tag_add("sel", "1.0", "end-1c")
        return "break"

    def copy_all(self) -> None:
        content = self.text.get("1.0", "end").strip()
        if not content:
            self._flash(t("ui.nothing_to_copy"), 1500)
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(content)
        self.root.update()   # без этого буфер обмена не успевает наполниться
        self._flash(t("ui.copied"), 1500)

    def clear_text(self) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self._flash(t("ui.cleared"), 2500)

    def _on_language(self, _event=None) -> None:
        code = language_code(self.language_var.get())
        if code == self.language.get():
            return

        self.language.set(code)
        self._save(language=code)
        # Фразы, уже стоящие в очереди, распознаются новым языком. Так и
        # надо: переключают язык ровно тогда, когда собеседник на него
        # перешёл, и хвост очереди относится уже к новой речи.
        if code is None:
            self._flash(t("ui.language_auto"), 7000)
        else:
            self._flash(t("ui.language_set", language=language_name(code)), 2500)

        # Снять фокус с выпадающего списка, иначе следующее нажатие
        # стрелки уедет в него, а не в текст.
        self.root.focus_set()

    def _on_ui_language(self, _event=None) -> None:
        code = ui_language_code(self.ui_language_var.get())
        if code == self._ui_language():
            return

        set_language(code)
        self._save(ui_language=code)
        self._apply_texts()
        self._flash(t("ui.interface_set", language=ui_language_name(code)), 2500)
        self.root.focus_set()

    def _on_model(self, _event=None) -> None:
        chosen = self.model_var.get()
        if chosen == t("ui.browse_model"):
            self._browse_model()
            return

        value = self._model_values.get(chosen)
        if value is None or value == (self._awaiting_model or self.model.get()):
            self._sync_model_var()
            return
        self._request_model(value)

    def _browse_model(self) -> None:
        path = filedialog.askdirectory(
            title=t("ui.model_dialog_title"), parent=self.root, mustexist=True
        )
        if not path:                       # отменили — вернуть прежний выбор
            self._sync_model_var()
            return

        path = str(Path(path))             # нормализуем разделители пути
        problem = model_problem(path)
        if problem is not None:
            key, params = problem
            self._flash(t("ui.model_rejected", reason=t(key, **params)), 9000)
            self._sync_model_var()
            return

        if self.store is not None:
            # Историю выбора запоминаем сразу, ещё до загрузки: она про то,
            # куда пользователь ходил, а не про то, что удалось поднять.
            self.store.data.remember_model(path)
        self._request_model(path)

    def _request_model(self, value: str) -> None:
        self.model.request(value)
        self._awaiting_model = value
        self._rebuild_model_choices()
        self._flash(t("ui.model_requested", model=model_display(value)), 6000)
        self.root.focus_set()

    def toggle_pause(self) -> None:
        if self.pause_event.is_set():
            self.pause_event.clear()
        else:
            self.pause_event.set()
        self.pause_button.configure(
            text=t("ui.resume") if self.pause_event.is_set() else t("ui.pause")
        )
        self._transient_until = 0.0
        self._refresh_status()

    def on_close(self) -> None:
        self._closing = True
        self.status_var.set(t("ui.stopping"))
        self.stop_event.set()
        self.pause_event.clear()
        self.root.update()
        # Tk теряет владение буфером обмена при выходе — небольшая пауза
        # даёт системе забрать содержимое, если пользователь только что
        # нажал «Скопировать всё».
        self.root.after(300, self.root.destroy)

    def run(self) -> None:
        self.root.mainloop()   # блокирует до закрытия окна
