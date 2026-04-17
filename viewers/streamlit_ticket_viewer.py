#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def discover_jsonl_files() -> List[str]:
    candidates = []
    for path in Path(".").glob("*.jsonl"):
        candidates.append(str(path))
    return sorted(candidates)


@st.cache_data
def load_tickets(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def ticket_ground_truth(ticket: Dict[str, Any]) -> Dict[str, Any]:
    gt = ticket.get("ground_truth")
    return gt if isinstance(gt, dict) else {}


def ticket_category(ticket: Dict[str, Any]) -> str:
    return str(ticket_ground_truth(ticket).get("primary_issue_category") or "unknown")


def ticket_flow(ticket: Dict[str, Any]) -> str:
    return str(ticket_ground_truth(ticket).get("flow_type") or "unknown")


def ticket_resolution_state(ticket: Dict[str, Any]) -> str:
    gt_state = str(ticket_ground_truth(ticket).get("resolution_state") or "").strip().lower()
    if gt_state:
        return gt_state
    status = str(ticket.get("status") or "").strip().lower()
    if status in {"closed", "resolved", "complete", "completed"}:
        return "resolved"
    if status in {"pending", "in progress"}:
        return "partial"
    return "unresolved"


def ticket_matches_search(ticket: Dict[str, Any], keyword: str) -> bool:
    if not keyword:
        return True
    needle = keyword.lower()
    haystacks = [
        str(ticket.get("ticket_id") or ""),
        str(ticket.get("title") or ""),
        str(ticket.get("description") or ""),
        ticket_category(ticket),
        ticket_flow(ticket),
        ticket_resolution_state(ticket),
    ]
    for message in ticket.get("messages") or []:
        haystacks.append(str(message.get("text") or ""))
        haystacks.append(str(message.get("author_name") or ""))
    return needle in "\n".join(haystacks).lower()


def filter_tickets(
    tickets: List[Dict[str, Any]],
    category_filter: str,
    resolution_filter: str,
    flow_filter: str,
    keyword: str,
) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for ticket in tickets:
        if category_filter != "All" and ticket_category(ticket) != category_filter:
            continue
        if resolution_filter != "All" and ticket_resolution_state(ticket) != resolution_filter:
            continue
        if flow_filter != "All" and ticket_flow(ticket) != flow_filter:
            continue
        if not ticket_matches_search(ticket, keyword):
            continue
        filtered.append(ticket)
    return filtered


def ticket_label(ticket: Dict[str, Any]) -> str:
    ticket_id = str(ticket.get("ticket_id") or "unknown-id")
    title = str(ticket.get("title") or "(untitled)")
    category = ticket_category(ticket)
    state = ticket_resolution_state(ticket)
    return f"{ticket_id} | {category} | {state} | {title}"


def render_message(message: Dict[str, Any]) -> None:
    role = str(message.get("author_role") or "unknown")
    author = html.escape(str(message.get("author_name") or "Unknown"))
    created_at = html.escape(str(message.get("created_at") or ""))
    text = html.escape(str(message.get("text") or "")).replace("\n", "<br>")

    if role == "requester":
        background = "#e8f1ff"
        border = "#8fb7ff"
        badge = "Requester"
    elif role == "agent":
        background = "#eef8ea"
        border = "#98c78a"
        badge = "Agent"
    else:
        background = "#f5f5f5"
        border = "#d0d0d0"
        badge = role.title() or "Unknown"

    st.markdown(
        f"""
        <div style="
            background:{background};
            border:1px solid {border};
            border-radius:10px;
            padding:12px;
            margin-bottom:10px;
        ">
            <div style="font-size:0.9rem; color:#333; margin-bottom:6px;">
                <strong>{badge}</strong> | {author}
                <span style="color:#666;">{created_at}</span>
            </div>
            <div style="font-size:1rem; line-height:1.45;">{text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_ticket_detail(ticket: Dict[str, Any]) -> None:
    gt = ticket_ground_truth(ticket)
    st.subheader(f"{ticket.get('ticket_id')} - {ticket.get('title')}")
    st.caption(
        f"Category: {ticket_category(ticket)} | "
        f"State: {ticket_resolution_state(ticket)} | "
        f"Flow: {ticket_flow(ticket)} | "
        f"Status: {ticket.get('status')}"
    )

    description = str(ticket.get("description") or "").strip()
    if description:
        st.markdown("**Description**")
        st.write(description)

    meta_cols = st.columns(3)
    meta_cols[0].metric("Messages", len(ticket.get("messages") or []))
    meta_cols[1].metric("Secondary Issue", str(gt.get("secondary_issue_category") or "-"))
    with meta_cols[2]:
        st.markdown("**Root Cause**")
        st.caption(str(gt.get("root_cause_id") or "-"))

    st.markdown("**Conversation Thread**")
    for message in ticket.get("messages") or []:
        if isinstance(message, dict):
            render_message(message)

    resolution_summary = gt.get("resolution_summary")
    if resolution_summary:
        st.markdown("**Ground Truth Resolution Summary**")
        st.info(str(resolution_summary))


def main() -> None:
    st.set_page_config(page_title="Simulated Ticket Viewer", layout="wide")
    st.title("Simulated Ticket Viewer")
    st.caption("Review generated tickets with filters and a readable conversation view.")

    discovered = discover_jsonl_files()
    default_path = "simulated_modular_200.jsonl" if "simulated_modular_200.jsonl" in discovered else (discovered[0] if discovered else "simulated_200.jsonl")

    with st.container():
        file_path = st.text_input("Ticket JSONL path", value=default_path)
        if discovered:
            selected = st.selectbox("Detected JSONL files", options=discovered, index=discovered.index(default_path) if default_path in discovered else 0)
            if selected != file_path:
                file_path = selected

    try:
        tickets = load_tickets(file_path)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    categories = ["All"] + sorted({ticket_category(ticket) for ticket in tickets})
    resolution_states = ["All"] + sorted({ticket_resolution_state(ticket) for ticket in tickets})
    flow_types = ["All"] + sorted({ticket_flow(ticket) for ticket in tickets})

    left_col, right_col = st.columns([1, 2], gap="large")

    with left_col:
        st.markdown("**Filters**")
        category_filter = st.selectbox("Issue Category", categories)
        resolution_filter = st.selectbox("Resolution State", resolution_states)
        flow_filter = st.selectbox("Flow Type", flow_types)
        keyword = st.text_input("Search by keyword", value="", placeholder="vpn, printer, token, urgent...")

        filtered = filter_tickets(tickets, category_filter, resolution_filter, flow_filter, keyword)
        st.caption(f"{len(filtered)} of {len(tickets)} tickets shown")

        if not filtered:
            st.warning("No tickets match the current filters.")
            selected_ticket = None
        else:
            labels = [ticket_label(ticket) for ticket in filtered]
            selected_label = st.radio("Ticket List", labels, label_visibility="collapsed")
            selected_ticket = filtered[labels.index(selected_label)]

    with right_col:
        if selected_ticket is None:
            st.info("Choose a ticket from the left to view its full thread.")
        else:
            render_ticket_detail(selected_ticket)


if __name__ == "__main__":
    main()

