# CallCribe

Live transcription of work calls on Windows, entirely on your own machine.

It listens to your microphone and to the system audio at the same time, cuts
speech into phrases at the pauses, transcribes with `faster-whisper`, and
shows the text in a window you can copy from. Every line is also written to a
Markdown file as it arrives.

**Nothing is sent anywhere.** No audio, no text ever leaves the machine. The
only network access in the app's whole life is the one-time model download on
first start, and you can [avoid even that](#fully-offline-install).

Interface in English or Russian; speech recognition in Russian, English or
Spanish. Those two are separate settings — an English interface transcribes a
Russian call just fine.

> Читаете по-русски? [README.ru.md](README.ru.md) — та же документация,
> подробнее в местах про русскую речь.

---

## Requirements

Read this bit. Most of what can go wrong is here.

| | |
|---|---|
| **OS** | Windows 10 or 11. Not portable — system audio is captured through WASAPI loopback, a Windows-only API. |
| **Python** | 3.10 – 3.12. 3.11 or 3.12 are the safe picks; the newest release often has no `PyAudioWPatch` wheel yet. The app refuses to start below 3.10 with a message saying so. |
| **Disk** | ~2 GB for the environment, plus the model: about 1.6 GB for `large-v3-turbo`, about 3 GB for `large-v3`. Models are cached in `%USERPROFILE%\.cache\huggingface`, not in this folder. |
| **Internet** | Once, to download the model. After that the app is fully offline — see [Fully offline install](#fully-offline-install) if it must never touch the network. |
| **GPU** | Optional. An NVIDIA card with a current driver gets you a ~25x real-time margin. Without one it runs on the CPU and automatically starts with a lighter model. No system-wide CUDA toolkit needed — the wheels carry their own DLLs. |
| **Headphones** | Strongly recommended. Over speakers the other party's voice also reaches your microphone and every phrase gets transcribed twice, once per channel. See [Limitations](#limitations). |

## Install

```powershell
git clone https://github.com/dx125/CallCribe.git
cd CallCribe
powershell -ExecutionPolicy Bypass -File setup.ps1
```

That is the whole thing. `setup.ps1` finds a suitable Python, creates `.venv`,
and installs dependencies — with the CUDA wheels if it sees an NVIDIA driver
and without them if it does not. Force the choice with `-Gpu` or `-Cpu` when
the guess is wrong.

<details>
<summary>Doing it by hand instead</summary>

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt        # CPU
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt    # + CUDA 12, ~700 MB
```

`requirements-gpu.txt` includes `requirements.txt`, so it is one command
either way, never both.
</details>

Check the machine before you rely on it in a real call:

```powershell
.venv\Scripts\python.exe selftest.py
```

It runs synthetic audio through the resampler and the VAD, exercises the
hallucination filter, verifies the message catalog against the code, saves and
reloads settings, switches models against a stub, then lists the audio devices
it found and says whether transcription will land on the GPU or the CPU. No
model needed. Add `--load-model` to also load whisper for real and transcribe
one phrase.

## Run

Create a desktop shortcut once:

```powershell
powershell -ExecutionPolicy Bypass -File install-shortcut.ps1
```

From then on, double-click **CallCribe** before the call starts.

```
CallCribe.lnk (desktop)  ->  pythonw.exe -m callcribe    no console, for everyday use
run.cmd                  ->  python.exe  -m callcribe    with a console, for diagnostics
```

The shortcut runs `pythonw.exe`, so there is **no console window**. That is
deliberate: a console sitting next to the transcript window is easy to close by
mistake, and closing it kills the process hard. With the shortcut there is one
window, and it shuts down cleanly.

**Use `run.cmd` for the first start** — the model download prints its progress
there, and a silent 3 GB wait is otherwise indistinguishable from a hang. Use
it any time something misbehaves and you want to see why.

Both accept flags that preset the window:

| Flag | Values | Sets |
|---|---|---|
| `--lang` | `ru` `en` `es` `auto` | Speech language |
| `--ui-lang` | `en` `ru` | Interface language |
| `--model` | a size name or a folder path | Whisper model |

None of them are required — everything is switchable inside the window. They
just supply a different starting value instead of the saved one, and behave
exactly like the dropdowns: they reach the settings file as soon as you change
anything in the window. A bad `--model` path is rejected immediately rather
than after half a minute of loading.

Transcripts are saved to `%USERPROFILE%\call-transcripts\call_YYYYMMDD_HHMMSS.md`.

### First start, step by step

1. The window opens right away and says `Loading the model...`.
2. `faster-whisper` downloads the weights — about 1.6 GB on a CPU machine
   (`large-v3-turbo`), about 3 GB with a GPU (`large-v3`). Once only; after
   that the model comes from the local cache.
3. The status line turns into `ready, listening (...)`, naming the model and
   whether it is on `cuda` or `cpu`.
4. Speak, or start the call. Text appears 1–3 seconds after each phrase ends.

### Fully offline install

If this machine must never reach the network, fetch the model elsewhere and
point CallCribe at the folder:

1. On a machine with internet, download a **CTranslate2** conversion — for
   example [`Systran/faster-whisper-large-v3`](https://huggingface.co/Systran/faster-whisper-large-v3)
   or the smaller [`deepdml/faster-whisper-large-v3-turbo-ct2`](https://huggingface.co/deepdml/faster-whisper-large-v3-turbo-ct2).
   You need `model.bin`, `config.json` and a tokenizer file.
2. Copy the folder across.
3. Pick it with **Browse...** in the **Model:** dropdown, or start with
   `run.cmd --model "D:\models\faster-whisper-large-v3"`.

The choice is remembered, so this is a one-time step. See
[Choosing a model](#choosing-a-model) for what makes a folder valid.

## Where your choices are stored

Three things you change in the window — interface language, speech language
and model — survive a restart. They live in

```
%APPDATA%\CallCribe\settings.json
```

The file is written immediately on each change, not at exit: the app runs for
a whole call and may not reach a clean shutdown, and losing the model choice
to that would be the most annoying way to lose it. The model is the exception
in one direction — it is saved only once it has **actually loaded**, so a
broken pick cannot survive a restart and turn the next launch into a failure.

A corrupt file breaks nothing: the app says so in the status line and falls
back to defaults. To return to what is written in `config.py`, delete it.

## How it works

```
Microphone                  System audio (loopback)
    │                                │
    ▼                                ▼
AudioCapture                    AudioCapture          one thread per source:
  reads WASAPI,                   reads WASAPI,       read, mix to mono,
  resamples to 16 kHz             resamples to 16 kHz resample, timestamp
    │                                │
    ▼                                ▼
VadSegmenter                    VadSegmenter          one thread per source:
  cuts phrases at pauses          cuts phrases        webrtcvad, preroll,
    │                                │                trailing-silence trim
    └────────────────┬───────────────┘
                     ▼
             shared phrase queue
                     ▼
           TranscriberWorker                          one thread per process:
      a single faster-whisper model,                  the queue itself
      hallucination filtering                         serializes model access
                     ▼
        ┌────────────┴────────────┐
        ▼                         ▼
  TranscriptWriter            GUI queue
  writes .md as it goes           ▼
                          TranscriptWindow            main thread:
                          Tkinter, copy               Tkinter requires it
```

| Module | Role |
|---|---|
| [config.py](callcribe/config.py) | Defaults in one dataclass, plus the language and model lists and the per-language prompts |
| [i18n.py](callcribe/i18n.py) | Message catalog: English and Russian interface |
| [settings.py](callcribe/settings.py) | What the user picked — read at start, written on every change |
| [audio.py](callcribe/audio.py) | Device selection and WASAPI capture |
| [resample.py](callcribe/resample.py) | Streaming anti-aliased resampling to 16 kHz |
| [vad.py](callcribe/vad.py) | Cutting the stream into phrases at pauses |
| [asr.py](callcribe/asr.py) | Loading whisper, transcription, CPU fallback |
| [filters.py](callcribe/filters.py) | Hallucination filtering |
| [transcript.py](callcribe/transcript.py) | Writing the `.md`, during and at the end |
| [ui.py](callcribe/ui.py) | The live text window |
| [app.py](callcribe/app.py) | Wiring and the main loop |

> **A note for contributors:** the interface and this documentation are
> bilingual, but the comments and docstrings inside the code are in Russian.
> They carry a lot of the reasoning — measurements, why a threshold is what it
> is, which failure a guard is for — so they are worth running through a
> translator rather than skipping.

## Configuration

Everything lives in [config.py](callcribe/config.py).

Three of the values — `ui_language`, `language` and `whisper_model` — are
**first-launch** defaults only. After that whatever you picked in the window
wins, from `settings.json` (see [Where your choices are stored](#where-your-choices-are-stored)).

| Setting | Default | Why change it |
|---|---|---|
| `whisper_device` | `auto` | Force `cuda` / `cpu`. `auto` takes the GPU when there is one and falls back to the CPU by itself |
| `whisper_model` | `large-v3` | First-launch model: a size name or a folder path. Changed live in the window, see [Choosing a model](#choosing-a-model) |
| `cpu_fallback_model` | `large-v3-turbo` | What to use **at startup** when there is no usable GPU. On the CPU everything is dominated by a fixed per-call cost — `large-v3` ~3.1 s, turbo ~2.6 s, with a six times smaller slope. On the 2–5 s phrases a conversation is made of, `large-v3` falls behind real time and turbo keeps up. This substitution never applies to a model you picked in the window — that one loads as asked |
| `ui_language` | `en` | First-launch interface language: `en` or `ru`. Unrelated to the speech language |
| `language` | `ru` | First-launch **speech** language — switchable live, see [Choosing a language](#choosing-a-language). `None` means auto-detect. **Change this if you do not transcribe Russian**, or just pick another one in the window |
| `restrict_auto_language` | `True` | In Auto mode, replace an off-list detected language with the nearest one on the list. Turn it off only if your calls contain something other than Russian, English and Spanish |
| `prompts` | a term list per language | The only terminology hint in v1. Keep your actual stack here — measured, it is the single most effective accuracy lever. The shipped lists are ASP.NET / EF Core terms; **replace them with yours**. Whisper silently truncates the prompt at 223 tokens, and `selftest` warns if you cross it |
| `vad_aggressiveness` | `2` (0–3) | Higher: fewer false triggers on noise, more risk of clipping quiet speech |
| `end_silence_ms` | `600` | How much silence ends a phrase |
| `soft_utterance_ms` | `8000` | How long a monologue must run before the segmenter starts looking for a pause to cut at. Lower: text appears more often, phrases fragment more |
| `max_utterance_ms` | `20000` | Hard cap. Never set above 30000 — whisper's window is exactly 30 s |
| `beam_size` | `5` | **Do not lower it.** Measured 30% WER at `beam_size=1` against 5.9%: the model does not degrade gracefully, it drops 20–30 second stretches whole |
| `show_speaker_labels` | `False` | `Me` / `Them` labels (or `Я` / `Собеседник`, following the interface language). Free — the two sources are genuinely separate channels |
| `output_dir` | `~/call-transcripts` | Where the `.md` files go |

## Choosing a language

The **Speech:** dropdown holds **Русский · English · Español · Auto**. It
switches during a call: the language is re-read for every phrase, so a change
takes effect on the very next one, including phrases already queued.

Language names are written in their own language and do not follow the
interface language — people scan a language list for a familiar shape, and
"Spanish" does not help with that. Only "Auto" is translated; it is a mode,
not a language.

Switching the language also switches the terminology hint: a list of Russian
words fed to English speech drags the output back into Russian, so `prompts`
holds a separate one per language.

**Auto is a second-class mode, and here is why.** Whisper detects the language
per phrase, not per call. On a one-to-two-second remark it will happily pick
Welsh or Korean out of its hundred languages — and then it *translates*
instead of transcribing. So:

- the set of expected languages is known, which makes the miss obvious: if the
  detector names something off-list, the most probable on-list language is used
  instead (`restrict_auto_language`). This is cheap — faster-whisper detects
  the language **before** it starts decoding, so the wrong pass is cancelled
  before it computes anything;
- the first time this happens the app says so in the status line — a hint that
  you should pick the call's language explicitly;
- when you know the language, an explicit pick always beats Auto. English
  technical terms inside Russian speech are recognized fine with "Русский".

Auto is for calls where you genuinely do not know what language will be
spoken, or where it changes partway through.

## Interface language

The **Interface:** dropdown holds **English · Русский**, English by default.
It is a completely separate setting from the speech language.

It switches live, with no restart — labels, buttons, status and messages are
all redrawn immediately.

The saved transcript is the interesting case. Speaker labels in it (`**Me**` /
`**Я**`) are stored as a channel marker rather than as text, and the file is
rebuilt whole on a clean close. So even if you switched language mid-call, the
resulting file comes out in one language — the one selected at the end.

Messages live in [i18n.py](callcribe/i18n.py) as a single key -> {en, ru}
dictionary. `selftest` checks that every key has both languages, that
placeholders (`{model}`, `{error}`) match across translations, and that the set
of keys in the catalog is exactly the set used in the code — catching both a
typo in a call and a forgotten translation.

Adding a third language is mechanical: add the code to `UI_LANGUAGES` and a
third value to each key in `MESSAGES`. `selftest` will list whatever you missed.

## Choosing a model

The **Model:** dropdown holds the names `faster-whisper` downloads itself,
plus **Browse...** for a model folder on disk.

```
tiny · base · small · medium · large-v2 · large-v3 · large-v3-turbo · distil-large-v3
```

Not any folder will do: it must be a model **converted to CTranslate2**, not
the original OpenAI `.pt`. The app checks the folder before loading and
requires `model.bin`, `config.json` and a tokenizer (`tokenizer.json`, or
`vocabulary.txt` / `vocabulary.json`). The tokenizer matters on its own:
without it `faster-whisper` silently goes to Hugging Face for one, and in an
offline app that is not an error, it is a hang. A ready example sits in the
cache already: `%USERPROFILE%\.cache\huggingface\hub\models--*\snapshots\*`.

Browsed folders are remembered — the last eight — and appear in the list next
time. The list shows the folder name rather than the full path, which would be
wider than the window; if two names collide, the parent folder disambiguates.

Switching models is not an assignment but tens of seconds of work, and the
**transcription thread** does it, not the window: a CTranslate2 model belongs
to the thread that created it, and touching it from outside crashes the process
(`0xC0000409`). The window files a request, the thread picks it up between
phrases. Hence the behavior:

- the current phrase finishes on the old model, the next one runs on the new;
- while it loads, the status line says `Loading ...` — otherwise tens of
  seconds of silence would look like a hang;
- the old model is released **before** the new one loads. Two `large-v3` at
  once is 3 GB extra, and on a mid-range card the switch simply would not fit;
- if the new one fails to load, the old one comes back and the call keeps
  being transcribed. That choice is not written to settings — otherwise the
  failure would survive a restart.

## What is done about hallucinations

Whisper was trained on YouTube subtitles and, given silence, confidently emits
pieces of that corpus: "Продолжение следует...", "Субтитры сделал DimaTorzok",
`Thanks for watching`, `Suscríbete al canal`. The clichés differ per language,
so the list covers all three. With two channels permanently open this is the
main source of garbage, so filtering happens in three layers:

1. **webrtcvad** — a segment never reaches the model at all if there is no
   speech in it.
2. **Thresholds inside faster-whisper** — `vad_filter`, `no_speech_threshold`,
   `log_prob_threshold`, `compression_ratio_threshold`, plus trimming trailing
   silence from the segment (a long silent tail provokes invention more than
   anything else).
3. **[filters.py](callcribe/filters.py)** — an explicit list of subtitle
   clichés and a decoder-loop detector. Looping comes in two sizes: by word
   ("да да да да да...") and by whole sentence, where whisper repeats a phrase
   verbatim two or three times. The `compression_ratio` threshold does not
   catch the second — measured 1.6 at two repeats and 2.4 at three against a
   threshold of 2.6 — so repeated runs are searched for explicitly. How many
   rounds are needed depends on the run's length: a short one people genuinely
   repeat ("one, two, three, one, two, three" is a real transcript line), so
   that needs four rounds, while a whole sentence twice verbatim is cut
   immediately. Across 896 lines of saved transcripts this rule did not touch
   a single genuine line.

If a cliché that is not on the list turns up in your transcript, add it to
`_EXACT_RAW` or `_MARKERS_RAW`. Keep markers (substrings) long: `subtítulo`
alone would also cut a real conversation about localization, whereas
`subtítulos realizados por` only catches the cliché.

## Long speech and performance

Measured on an RTX 4070 Ti, `large-v3`/`float16`/`beam=5`, live Russian speech:

| Fragment length | Transcription time | RTF |
|---|---|---|
| 1 s | 0.17 s | 0.17 |
| 5 s | 0.34 s | 0.07 |
| 20 s | 0.88 s | 0.04 |
| 45 s | 1.88 s | 0.04 |

That is **a fixed ~0.17 s per call plus ~0.034 s per second of audio**. The
fixed part is the encoder: whisper always works on a 30-second window and pads
the input with silence, so a one-second remark costs nearly as much as a
twenty-second one.

Two consequences:

- **A 25x speed margin.** The latency bottleneck is not computation but
  `end_silence_ms` (0.6 s waiting for the phrase to end) plus the time to a cut
  in a long monologue. A faster model or more VRAM does not change that.
- **Cutting long speech up helps accuracy too.** WER against a reference
  (102 s of Russian technical speech, 170 words): the whole file at once 5.9%,
  cut into phrases 4.1%. With `condition_on_previous_text=False` short pieces
  do not drag an error along, and each gets the `initial_prompt` terms afresh.

What mattered in that measurement and what did not:

| Change | WER | Verdict |
|---|---|---|
| phrase segmentation instead of the whole file | 5.9% -> 4.1% | clearly helps |
| `initial_prompt` with terms off -> on | 7.1% -> 5.9% | the most accessible lever |
| `beam_size` 5 -> 1 | 5.9% -> **30.0%** | never do this |
| `beam_size` 5 -> 10 | 5.9% -> 5.9% | pointless |
| `large-v3` -> `large-v3-turbo` (segmented) | 4.1% -> 4.1% | a tie on this passage, three times faster |

The collapse at `beam_size=1` is not graceful degradation: the model loses
20–30 second stretches whole.

Given a 25x speed margin, it follows that changing the model or adding VRAM is
pointless. The one lever that actually moves accuracy is keeping the stack you
really talk about in `initial_prompt`.

What happens to a monologue longer than 30 seconds:

1. It never reaches whisper in that form. `end_silence_ms` ends a phrase at any
   pause of 0.6 s — in live speech, roughly every 12 seconds.
2. If someone talks without such pauses, after `soft_utterance_ms` (8 s) the
   segmenter finds the **quietest moment** in the buffer and cuts through its
   middle, so the seam falls between words rather than inside one.
3. If there are no pauses at all (reading a list aloud), `max_utterance_ms`
   (20 s) applies a hard cut with a 200 ms overlap so the word at the seam is
   not lost.

**Memory will not run away.** A phrase buffer is bounded by
`max_utterance_ms` by construction: 20 s at 16 kHz float32 is 1.28 MB, no
matter how long someone talks. Capture, segmentation and transcription live in
different threads, so the system keeps listening while it computes the previous
chunk. The only thing that can grow without bound is the phrase queue, if
transcription turns out slower than real time — a 25x margin on the GPU, and on
the CPU the app warns in the status line.

## Limitations

- **1–3 second latency** after a phrase ends. Whisper is architecturally not
  streaming: segmentation follows pauses, not words. Word-by-word subtitles
  need a fundamentally different model (NVIDIA Parakeet with cache-aware
  streaming, for instance).
- **Headphones needed.** Over speakers the other party's voice also reaches
  your microphone, and each of their phrases is transcribed twice, once per
  channel. There is no echo cancellation in v1. This is **the most common cause
  of "the text is doubled"**: two lines with the same timestamp and nearly the
  same text. With `show_speaker_labels=False` they are indistinguishable in the
  file and look like a bug; turn the labels on and you will see it is `Me` and
  `Them`, i.e. echo.
- **Loopback captures all system audio**, not just the call: music, a browser
  video, notifications — everything lands in the transcript. That is what the
  **Pause** button is for.
- **No diarization.** In a group call the entire far side is one `Them`
  channel, with no separation into individual participants.
- **One language per phrase.** The dropdown switches recognition as a whole; a
  phrase where someone switches from Russian to English mid-sentence is parsed
  as monolingual. Auto does not help — it detects per phrase, not per part.
- **Bluetooth headsets in hands-free mode (HFP)** drop the input sample rate to
  8 kHz; recognition of the other party will noticeably suffer.
- **Exclusive-mode output devices.** If an application has grabbed the output
  exclusively, loopback will not work — the app shows a clear error at startup.
- **The microphone is optional**: without one the app starts anyway and records
  only the other side, with a warning.
- **Clipboard and closing the window.** Tk loses ownership of the clipboard on
  exit, so paste what "Copy all" gave you before closing the window. The saved
  file is unaffected — it is written independently.
- **Close the transcript window, not the console.** The console next to it is
  the log. Closing that one kills the process hard (CTranslate2 carries an
  Intel runtime with its own console-close handler) and the file is not
  re-sorted by time. The lines are on disk regardless — they are written as
  they arrive.

## Saving

Lines are written to the `.md` **as they are transcribed**, not at the end: a
crash or a kill should not eat a forty-minute call. On a clean window close the
file is rewritten, sorted by phrase start time — two independent channels
arrive interleaved.

## Troubleshooting

**`setup.ps1` will not run: "running scripts is disabled on this system".**
That is PowerShell's default execution policy. The command in
[Install](#install) already works around it with `-ExecutionPolicy Bypass`,
which applies to that one invocation and changes nothing permanently.

**"No Python 3.10 or newer found", but Python is installed.** Most likely
`python` is the Microsoft Store stub, which does nothing but open the Store.
Install from [python.org](https://www.python.org/downloads/windows/) with
"Add python.exe to PATH" ticked, or check `py --list`.

**"Loopback device not found" at startup.** WASAPI loopback attaches to the
*current default output device*. Play something to confirm you know which
device that is, and check Windows Settings → System → Sound. If an application
holds the output in exclusive mode, release it and restart CallCribe.

**"CUDA compute libraries are missing (cublas64_12.dll, ...)".** You have an
NVIDIA GPU and its driver, but not the CUDA wheels — most often from installing
`requirements.txt` instead of `requirements-gpu.txt` on a GPU machine. The
driver alone is enough for the GPU to be *visible* but not to compute on, so
CallCribe checks for the compute libraries up front and transcribes on the CPU
instead. Nothing is broken; the card is just idle. To use it:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
```

`selftest.py` reports this too, under **CUDA**.

**The window says it is falling behind, and the queue keeps growing.** You are
on the CPU with too heavy a model. Pick `large-v3-turbo` or smaller in the
**Model:** dropdown. From a cold start the app already does this for you; it
does not second-guess a model you chose yourself.

**Text appears doubled, two nearly identical lines.** Speaker echo through
speakers. Use headphones. Set `show_speaker_labels = True` in `config.py` to
confirm — you will see `Me` and `Them` rather than one line twice.

**It transcribes music and browser videos.** Loopback takes everything the
system plays. Use **Pause**.

**Nothing at all appears, no errors.** Run `run.cmd` instead of the shortcut
and read the console; then `selftest.py`, which will tell you whether the
devices were found and where the model landed.

## Roadmap (v2+)

- Term normalization with a small local LLM (LM Studio / `llama-server`) as
  post-processing on the saved file, leaving live latency untouched.
- `show_speaker_labels=True` with visual styling (different colors for `Me`
  and `Them`).
- Replacing pause-based segmentation with a streaming model for word-by-word
  subtitles.
- An "Export" button with a format choice (`.txt`, `.docx`).
- Echo cancellation, so speakers become usable.
- More interface languages.

## License

MIT — see [LICENSE](LICENSE).
