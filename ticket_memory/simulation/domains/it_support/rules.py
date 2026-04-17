from __future__ import annotations

from collections import Counter
from typing import Dict

from ticket_memory.simulation.domains.base import DomainPack

from .catalog import IT_SUPPORT_VARIANTS


def build_it_support_domain() -> DomainPack:
    counts = Counter(variant.family for variant in IT_SUPPORT_VARIANTS)
    family_weights: Dict[str, float] = {family: float(count) for family, count in counts.items()}
    return DomainPack(
        name="it_support",
        variants=IT_SUPPORT_VARIANTS,
        flow_weights={
            "direct_resolution": 0.22,
            "failed_first_fix_then_success": 0.31,
            "escalation": 0.13,
            "partial_resolution": 0.14,
            "mixed_issue": 0.20,
        },
        family_weights=family_weights,
    )


