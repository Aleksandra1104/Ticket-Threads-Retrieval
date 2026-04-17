from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class ExtractedPairRecord:
    ticket_id: str
    query: str
    positive: str
    issue_category: str
    confidence: float
    reasoning_short: str
    used_message_indexes: List[int]
    title: str
    status: str
    metadata: Dict[str, Any]


