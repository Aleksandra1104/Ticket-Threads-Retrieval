from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ticket_memory.simulation.core.models import ConversationResult
from ticket_memory.simulation.core.utils import write_jsonl


def export_raw_threads(results: Sequence[ConversationResult], output_path: Path) -> None:
    write_jsonl(output_path, [result.to_ticket_dict() for result in results])


