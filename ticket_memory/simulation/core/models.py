from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Message:
    author_name: str
    author_role: str
    created_at: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Persona:
    name: str
    style: str
    prefixes: List[str] = field(default_factory=list)
    suffixes: List[str] = field(default_factory=list)
    tendencies: List[str] = field(default_factory=list)


@dataclass
class IssueVariant:
    family: str
    root_cause_id: str
    titles: List[str]
    user_openers: List[str]
    clarify_agent: List[str]
    clarify_user: List[str]
    first_try_agent: List[str]
    first_try_user_fail: List[str]
    diagnosis_agent: List[str]
    fix_agent: List[str]
    resolution_summary: List[str]
    environment_details: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


@dataclass
class Scenario:
    ticket_id: str
    domain: str
    issue_variant: IssueVariant
    title: str
    status: str
    flow_type: str
    user_persona: Persona
    agent_persona: Persona
    requester_name: str
    agent_name: str
    include_secondary_issue: bool = False
    secondary_issue_variant: Optional[IssueVariant] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationResult:
    scenario: Scenario
    description: str
    messages: List[Message]
    resolution_summary: Optional[str]
    resolution_state: str
    should_extract: bool
    confidence: float

    def to_ticket_dict(self) -> Dict[str, Any]:
        secondary = None
        if self.scenario.secondary_issue_variant is not None:
            secondary = self.scenario.secondary_issue_variant.family

        return {
            "ticket_id": self.scenario.ticket_id,
            "status": self.scenario.status,
            "title": self.scenario.title,
            "description": self.description,
            "messages": [message.to_dict() for message in self.messages],
            "ground_truth": {
                "primary_issue_category": self.scenario.issue_variant.family,
                "secondary_issue_category": secondary,
                "resolution_summary": self.resolution_summary,
                "should_extract": self.should_extract,
                "resolution_state": self.resolution_state,
                "flow_type": self.scenario.flow_type,
                "root_cause_id": self.scenario.issue_variant.root_cause_id,
                "domain": self.scenario.domain,
                "user_persona": self.scenario.user_persona.name,
                "agent_persona": self.scenario.agent_persona.name,
                "metadata": self.scenario.metadata,
            },
        }


