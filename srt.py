"""Build sentence-level subtitle segments and an SRT string from chunk durations."""


def build_segments(texts, durations, gap: float = 0.0) -> list[dict]:
    if len(texts) != len(durations):
        raise ValueError(f"texts ({len(texts)}) and durations ({len(durations)}) must match")
    segments, cursor = [], 0.0
    for i, (text, dur) in enumerate(zip(texts, durations), start=1):
        start = cursor
        end = start + float(dur)
        segments.append({"index": i, "start": round(start, 3), "end": round(end, 3), "text": text})
        cursor = end + gap
    return segments


def _fmt(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def segments_to_srt(segments) -> str:
    blocks = []
    for seg in segments:
        blocks.append(f"{seg['index']}\n{_fmt(seg['start'])} --> {_fmt(seg['end'])}\n{seg['text']}\n")
    return "\n".join(blocks)
