from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, TypeVar

T = TypeVar("T")


def choose(seq: Sequence[T], rng: random.Random) -> T:
    return rng.choice(list(seq))


def maybe(seq: Sequence[T], rng: random.Random, probability: float) -> Optional[T]:
    if seq and rng.random() < probability:
        return choose(seq, rng)
    return None


def iso_time(ticket_index: int, step: int) -> str:
    day = 1 + (ticket_index % 27)
    base_minutes = (8 + (ticket_index % 5)) * 60 + ((11 * ticket_index) % 45)
    total_minutes = base_minutes + (step * 17)
    hour = (total_minutes // 60) % 24
    minute = total_minutes % 60
    return f"2026-02-{day:02d}T{hour:02d}:{minute:02d}:00Z"


def build_description(opener: str, title: str, supporting_messages: Sequence[str]) -> str:
    extra = " ".join(text for text in supporting_messages[:2] if text)
    parts = [title.strip(), f"Reported problem: {opener.strip()}"]
    if extra:
        parts.append(f"Context: {extra.strip()}")
    return " ".join(parts).strip()


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

