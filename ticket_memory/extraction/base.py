from __future__ import annotations

from typing import Any, Dict, Protocol


class ThreadPairExtractor(Protocol):
    def extract_pair(self, thread_text: str) -> Dict[str, Any]:
        ...

