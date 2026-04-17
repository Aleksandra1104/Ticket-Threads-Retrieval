from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from ticket_memory.simulation.core.models import IssueVariant


@dataclass
class DomainPack:
    name: str
    variants: List[IssueVariant]
    flow_weights: Dict[str, float]
    family_weights: Dict[str, float] = field(default_factory=dict)


