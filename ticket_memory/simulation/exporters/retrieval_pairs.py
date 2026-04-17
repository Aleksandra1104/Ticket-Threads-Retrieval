from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

from ticket_memory.simulation.core.models import ConversationResult
from ticket_memory.simulation.core.utils import write_jsonl


def build_query_text(result: ConversationResult) -> str:
    requester_messages = []
    for message in result.messages:
        text = message.text.strip()
        if message.author_role != "requester" or not text:
            continue
        lowered = text.lower()
        if (
            "it works now" in lowered
            or "that fixed it" in lowered
            or "looks good now" in lowered
            or "i can access it now" in lowered
        ):
            continue
        requester_messages.append(text)
    if requester_messages:
        return " ".join(requester_messages[:3]).strip()
    return result.description.strip()


def result_to_pair(result: ConversationResult) -> Dict[str, object]:
    ticket = result.to_ticket_dict()
    ground_truth = ticket["ground_truth"]
    return {
        "ticket_id": ticket["ticket_id"],
        "query": build_query_text(result),
        "positive": result.resolution_summary or "",
        "issue_category": ground_truth["primary_issue_category"],
        "confidence": result.confidence,
        "reasoning_short": f"Simulated from flow={ground_truth['flow_type']} root_cause={ground_truth['root_cause_id']}",
        "used_message_indexes": [index for index, message in enumerate(result.messages) if message.author_role == "requester"][:3],
        "title": ticket["title"],
        "status": ticket["status"],
        "ground_truth_primary_issue_category": ground_truth["primary_issue_category"],
        "ground_truth_secondary_issue_category": ground_truth["secondary_issue_category"],
        "ground_truth_resolution_state": ground_truth["resolution_state"],
        "metadata": {
            "domain": ground_truth["domain"],
            "flow_type": ground_truth["flow_type"],
            "root_cause_id": ground_truth["root_cause_id"],
            "user_persona": ground_truth["user_persona"],
            "agent_persona": ground_truth["agent_persona"],
            "ground_truth_primary_issue_category": ground_truth["primary_issue_category"],
            "ground_truth_secondary_issue_category": ground_truth["secondary_issue_category"],
            "ground_truth_resolution_state": ground_truth["resolution_state"],
        },
    }


def export_retrieval_pairs(results: Sequence[ConversationResult], output_path: Path, resolved_only: bool = True) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for result in results:
        if resolved_only and not result.should_extract:
            continue
        rows.append(result_to_pair(result))
    write_jsonl(output_path, rows)
    return rows

