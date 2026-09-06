"""Отсев галлюцинаций whisper.

Модель обучалась на субтитрах с YouTube, поэтому на тишине, дыхании,
щелчках клавиатуры и обрывках звука она уверенно выдаёт куски из этого
корпуса. При двух постоянно открытых каналах это самый заметный источник
мусора в расшифровке, и никакой порог вероятности его полностью не
закрывает — нужен явный список.
"""

from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

_PUNCT = re.compile(r"[^\w\s.]", re.UNICODE)
_SPACES = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    text = _PUNCT.sub(" ", text)
    text = text.replace(".", " ")
    return _SPACES.sub(" ", text).strip()


# Фразы целиком. Записаны как есть, а сравниваются в нормализованном виде —
# отсюда прогон через normalize() ниже (иначе, например, "amara.org"
# никогда не совпадёт с нормализованным "amara org").
_EXACT_RAW = {
    # русские субтитровые штампы
    "продолжение следует",
    "продолжение следует...",
    "субтитры сделал dimatorzok",
    "субтитры создавал dimatorzok",
    "субтитры делал dimatorzok",
    "редактор субтитров а синецкая корректор а егорова",
    "спасибо за просмотр",
    "спасибо за внимание",
    "спасибо за просмотр и до новых встреч",
    "подписывайтесь на канал",
    "подписывайтесь на наш канал",
    "ставьте лайки и подписывайтесь на канал",
    "не забудьте подписаться на канал",
    "всем пока",
    "до новых встреч",
    "продолжение в следующем видео",
    "музыка",
    "аплодисменты",
    "смех",
    # английские
    "thank you",
    "thank you.",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "subtitles by the amara.org community",
    "transcription by castingwords",
    "bye",
    "you",
    "the end",
    "silence",
    # испанские
    "gracias por ver el video",
    "gracias por ver el vídeo",
    "gracias por vernos",
    "gracias por su atención",
    "gracias por ver este video",
    "suscríbete al canal",
    "suscríbanse al canal",
    "no olvides suscribirte",
    "no olviden suscribirse",
    "hasta la próxima",
    "nos vemos en el próximo video",
    "subtítulos realizados por la comunidad de amara.org",
    "música",
    "aplausos",
    "risas",
    "silencio",
}

# Достаточно вхождения подстроки — это уже гарантированный мусор.
_MARKERS_RAW = (
    "dimatorzok",
    "amara.org",
    "castingwords",
    "субтитры сделал",
    "субтитры создавал",
    "редактор субтитров",
    "подписывайтесь на",
    "субтитр",
    # Испанские берём длинными: "subtítulo" одним словом отсекать нельзя —
    # в разговоре про локализацию оно вполне может прозвучать всерьёз.
    "subtítulos realizados por",
    "subtítulos por la comunidad",
    "subtitulado por",
    "suscríbete al canal",
    "suscríbanse al canal",
)

_EXACT = frozenset(normalize(phrase) for phrase in _EXACT_RAW)
_MARKERS = tuple(normalize(marker) for marker in _MARKERS_RAW)


def _is_degenerate(words: list[str]) -> bool:
    """Зацикливание декодера: одно слово или целая фраза по кругу."""
    if len(words) < 4:
        return False
    counts = Counter(words)
    top_word, top_count = counts.most_common(1)[0]
    if top_count >= 4 and top_count / len(words) > 0.6:
        return True

    for period in range(1, len(words) // 2 + 1):
        # Хвост может быть неполным: декодер обрывается там, где его
        # остановили, а не на границе оборота.
        if any(words[i] != words[i % period] for i in range(period, len(words))):
            continue
        times = len(words) / period
        # Короткий оборот человек повторяет и всерьёз — "раз, два, три,
        # раз, два, три" встретилось в реальной расшифровке, — поэтому для
        # него порог прежний: четыре круга. А вот целое предложение
        # дословно дважды подряд не говорит уже никто: сжатие такой строки
        # до порога compression_ratio не дотягивает (замерено: 1.6 при
        # двух повторах, 2.4 при трёх, порог 2.6), и до этой проверки она
        # доходила нетронутой.
        if times >= 4 or (period >= 4 and times >= 2):
            return True
    return False


def _compression_ratio(text: str) -> float:
    """Тот же показатель, что whisper использует внутри: сильно сжимаемый
    текст — почти всегда повторяющийся мусор."""
    data = text.encode("utf-8")
    if not data:
        return 0.0
    return len(data) / len(zlib.compress(data))


def is_hallucination(text: str, *, max_compression_ratio: float = 2.6) -> bool:
    """True, если строку не стоит показывать пользователю."""
    stripped = text.strip()
    if not stripped:
        return True

    norm = normalize(stripped)
    if not norm:
        return True

    if norm in _EXACT:
        return True
    if any(marker in norm for marker in _MARKERS):
        return True

    words = norm.split()

    # Одиночное междометие/артефакт длиной в один-два символа.
    if len(words) == 1 and len(words[0]) <= 2:
        return True

    if _is_degenerate(words):
        return True

    if len(stripped) > 40 and _compression_ratio(stripped) > max_compression_ratio:
        return True

    return False


def clean(text: str) -> str:
    """Косметика перед выводом: схлопывание пробелов и обрезка кавычек."""
    return _SPACES.sub(" ", text).strip().strip('"«»')
