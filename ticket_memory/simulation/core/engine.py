from __future__ import annotations

import random
from typing import List

from ticket_memory.simulation.domains.base import DomainPack
from ticket_memory.simulation.domains.it_support.artifacts import AGENT_NAMES, USER_NAMES

from .flows import FLOW_HANDLERS
from .models import ConversationResult, IssueVariant, Scenario
from .personas import AGENT_PERSONAS, USER_PERSONAS
from .utils import choose


class SimulationEngine:
    def __init__(self, domain_pack: DomainPack, rng: random.Random) -> None:
        self.domain_pack = domain_pack
        self.rng = rng

    def _weighted_flow(self) -> str:
        names = list(self.domain_pack.flow_weights.keys())
        weights = list(self.domain_pack.flow_weights.values())
        return self.rng.choices(names, weights=weights, k=1)[0]

    def _choose_variant(self) -> IssueVariant:
        return choose(self.domain_pack.variants, self.rng)

    def build_scenario(self, ticket_index: int) -> Scenario:
        issue_variant = self._choose_variant()
        flow_type = self._weighted_flow()
        secondary = None
        if flow_type == "mixed_issue":
            candidates = [variant for variant in self.domain_pack.variants if variant.family != issue_variant.family]
            secondary = choose(candidates, self.rng)

        status_by_flow = {
            "direct_resolution": "Closed",
            "failed_first_fix_then_success": "Closed",
            "mixed_issue": "Closed",
            "escalation": choose(["Open", "In Progress", "Pending"], self.rng),
            "partial_resolution": choose(["In Progress", "Pending"], self.rng),
        }
        return Scenario(
            ticket_id=f"TDX-{10000 + ticket_index}",
            domain=self.domain_pack.name,
            issue_variant=issue_variant,
            title=choose(issue_variant.titles, self.rng),
            status=status_by_flow[flow_type],
            flow_type=flow_type,
            user_persona=choose(USER_PERSONAS, self.rng),
            agent_persona=choose(AGENT_PERSONAS, self.rng),
            requester_name=choose(USER_NAMES, self.rng),
            agent_name=choose(AGENT_NAMES, self.rng),
            include_secondary_issue=secondary is not None,
            secondary_issue_variant=secondary,
            metadata={"ticket_index": ticket_index},
        )

    def generate_ticket(self, ticket_index: int) -> ConversationResult:
        scenario = self.build_scenario(ticket_index)
        return FLOW_HANDLERS[scenario.flow_type](scenario, self.rng)

    def generate_tickets(self, count: int) -> List[ConversationResult]:
        return [self.generate_ticket(index) for index in range(1, count + 1)]


