from __future__ import annotations

import random
from typing import Dict

from .models import Persona


TYPO_REPLACEMENTS: Dict[str, str] = {
    "cannot": "cant",
    "can't": "cant",
    "password": "passwrod",
    "account": "acount",
    "email": "emial",
    "printer": "prnter",
    "connection": "conection",
    "access": "acess",
    "please": "pls",
    "thanks": "thx",
    "issue": "isue",
    "trying": "tryng",
    "problem": "probelm",
    "shared": "shard",
    "folder": "foler",
    "locked": "lockd",
}


def maybe_typo_word(word: str, rng: random.Random, typo_rate: float = 0.14) -> str:
    stripped = word.strip(".,!?;:")
    lowered = stripped.lower()
    if lowered not in TYPO_REPLACEMENTS or rng.random() >= typo_rate:
        return word
    replacement = TYPO_REPLACEMENTS[lowered]
    if stripped.istitle():
        replacement = replacement.capitalize()
    elif stripped.isupper():
        replacement = replacement.upper()
    suffix = word[len(stripped):] if word.startswith(stripped) else ""
    if word and word[-1] in ".,!?;:":
        suffix = word[-1]
    return replacement + suffix


def apply_persona(text: str, persona: Persona, rng: random.Random, allow_typos: bool) -> str:
    result = text.strip()
    if persona.prefixes and rng.random() < 0.22:
        prefix = rng.choice(persona.prefixes).strip()
        if prefix:
            result = f"{prefix} {result}"
    if persona.suffixes and rng.random() < 0.22:
        suffix = rng.choice(persona.suffixes)
        if suffix:
            result = f"{result}{suffix}"
    if allow_typos and rng.random() < 0.40:
        words = [maybe_typo_word(word, rng) for word in result.split()]
        result = " ".join(words)
    if allow_typos and rng.random() < 0.08:
        result = result.lower()
    return result.strip()

