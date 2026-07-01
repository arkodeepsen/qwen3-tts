"""Split text into short generation units for stable long-form synthesis."""
import re

_TERMINATORS = ".!?。！？\n"
_SPLIT_RE = re.compile(r"[^.!?。！？\n]*[.!?。！？\n]+|[^.!?。！？\n]+$")


def _hard_split(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    words, out, cur = sentence.split(" "), [], ""
    for w in words:
        candidate = w if not cur else cur + " " + w
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            # a single word longer than max_chars: slice it
            while len(w) > max_chars:
                out.append(w[:max_chars]); w = w[max_chars:]
            cur = w
    if cur:
        out.append(cur)
    return out


def split_sentences(text: str, max_chars: int = 200) -> list[str]:
    pieces = _SPLIT_RE.findall(text or "")
    out: list[str] = []
    for p in pieces:
        s = p.strip().strip("\n").strip()
        if not s or all(ch in _TERMINATORS + " " for ch in s):
            continue
        out.extend(_hard_split(s, max_chars))
    return out


def pack_sentences(sentences: list[str], max_chars: int = 200) -> list[str]:
    out: list[str] = []
    cur = ""
    for s in sentences:
        candidate = s if not cur else cur + " " + s
        if len(candidate) <= max_chars:
            cur = candidate
        else:
            if cur:
                out.append(cur)
            cur = s
    if cur:
        out.append(cur)
    return out
