#!/usr/bin/env python3
from __future__ import annotations

import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from extraction.extract_ticket_pairs import Ticket, dict_to_ticket, load_tickets
from retrieval.answer_new_tickets import build_retrieval_query, load_index, retrieve_matches


def discover_paths(pattern: str) -> List[str]:
    return sorted(str(path) for path in Path(".").glob(pattern))


@st.cache_resource
def load_model(model_name_or_path: str) -> SentenceTransformer:
    return SentenceTransformer(model_name_or_path)


@st.cache_data
def load_ticket_file(path_str: str) -> List[Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return [ticket.raw for ticket in load_tickets(path)]


@st.cache_data
def load_history_ticket_map(path_str: str) -> Dict[str, Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    mapping: Dict[str, Dict[str, Any]] = {}
    for ticket in load_tickets(path):
        mapping[ticket.ticket_id] = ticket.raw
    return mapping


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


def render_ticket_thread(ticket: Dict[str, Any], show_ground_truth: bool = True) -> None:
    st.markdown(f"**{ticket.get('ticket_id', 'Unknown')} - {ticket.get('title', '(untitled)')}**")
    st.caption(f"Status: {ticket.get('status', '')}")

    description = str(ticket.get("description") or "").strip()
    if description:
        st.write(description)

    for message in ticket.get("messages") or []:
        if isinstance(message, dict):
            render_message(message)

    if show_ground_truth:
        gt = ticket.get("ground_truth")
        if isinstance(gt, dict) and gt.get("resolution_summary"):
            st.info(str(gt.get("resolution_summary")))


def parse_ticket_json(raw_text: str) -> Tuple[Optional[Ticket], Optional[str]]:
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"
    if not isinstance(payload, dict):
        return None, "Expected a single JSON object for one ticket."
    return dict_to_ticket(payload), None


def default_ticket_json() -> str:
    example_path = Path("example.json")
    if example_path.exists():
        return example_path.read_text(encoding="utf-8")
    return json.dumps(
        {
            "ticket_id": "TDX-NEW-001",
            "status": "new",
            "title": "New support issue",
            "description": "Describe the new issue here.",
            "messages": [
                {
                    "author_name": "Requester",
                    "author_role": "requester",
                    "created_at": "2026-04-11T09:00:00Z",
                    "text": "I need help with a new problem.",
                }
            ],
        },
        indent=2,
    )


def main() -> None:
    st.set_page_config(page_title="Ticket Retrieval Viewer", layout="wide")
    st.title("Ticket Retrieval Viewer")
    st.caption("Embed a new ticket, retrieve top-k historical matches, and inspect the retrieved threads.")

    model_candidates = ["models/ticket-pairs"]
    if Path("models").exists():
        model_candidates = sorted(str(path) for path in Path("models").iterdir() if path.is_dir()) or model_candidates

    index_candidates = [path for path in discover_paths("retrieval_index*") if Path(path).is_dir()]
    history_candidates = discover_paths("*.jsonl")

    with st.sidebar:
        st.markdown("**Retrieval Settings**")
        model_path = st.selectbox("Embedding model", options=model_candidates, index=0 if model_candidates else None)
        index_dir = st.selectbox("Index directory", options=index_candidates, index=0 if index_candidates else None)
        history_path = st.selectbox("Historical threads JSONL", options=["(none)"] + history_candidates, index=0)
        top_k = st.slider("Top-k matches", min_value=1, max_value=10, value=3)
        min_score = st.slider("Highlight threshold", min_value=0.0, max_value=1.0, value=0.55, step=0.01)

    source_tab, results_tab = st.tabs(["New Ticket", "Retrieved Matches"])

    with source_tab:
        input_mode = st.radio("Input mode", options=["Paste JSON", "Load from file"], horizontal=True)

        ticket: Optional[Ticket] = None
        parse_error: Optional[str] = None

        if input_mode == "Paste JSON":
            raw_json = st.text_area("New ticket JSON", value=default_ticket_json(), height=320)
            ticket, parse_error = parse_ticket_json(raw_json)
        else:
            candidate_files = history_candidates or ["example.json"]
            file_path = st.selectbox("Input ticket file", options=candidate_files)
            try:
                raw_tickets = load_ticket_file(file_path)
            except Exception as exc:
                parse_error = str(exc)
                raw_tickets = []
            if raw_tickets:
                labels = [f"{row.get('ticket_id', 'unknown')} | {row.get('title', '(untitled)')}" for row in raw_tickets]
                selected_label = st.selectbox("Ticket in file", options=labels)
                selected_row = raw_tickets[labels.index(selected_label)]
                ticket = dict_to_ticket(selected_row)

        if parse_error:
            st.error(parse_error)
        elif ticket is not None:
            st.markdown("**New Ticket Preview**")
            render_ticket_thread(ticket.raw, show_ground_truth=False)

        run_retrieval = st.button("Retrieve Similar Tickets", type="primary", use_container_width=True)

    with results_tab:
        if not run_retrieval:
            st.info("Load or paste a ticket, then click 'Retrieve Similar Tickets'.")
            return

        if ticket is None or parse_error:
            st.warning("Fix the ticket input before running retrieval.")
            return

        if not model_path or not index_dir:
            st.error("Model path and index directory are required.")
            return

        try:
            model = load_model(model_path)
            records, embeddings = load_index(Path(index_dir))
        except Exception as exc:
            st.error(f"Failed to load model or index: {exc}")
            return

        history_map: Dict[str, Dict[str, Any]] = {}
        if history_path != "(none)":
            try:
                history_map = load_history_ticket_map(history_path)
            except Exception as exc:
                st.warning(f"Could not load historical threads file: {exc}")

        retrieval_query = build_retrieval_query(ticket)
        query_embedding = model.encode(
            [retrieval_query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        matches = retrieve_matches(
            query_embedding=query_embedding,
            records=records,
            embeddings=embeddings,
            top_k=top_k,
        )

        st.markdown("**Retrieval Query**")
        st.code(retrieval_query)

        if not matches:
            st.warning("No matches found.")
            return

        st.markdown("**Top Matches**")
        selected_match_id = st.radio(
            "Retrieved Tickets",
            options=[match["ticket_id"] for match in matches],
            format_func=lambda ticket_id: next(
                (
                    f"{match['ticket_id']} | score={match['score']} | {match.get('issue_category', 'unknown')} | {match.get('title', '(untitled)')}"
                    for match in matches
                    if match["ticket_id"] == ticket_id
                ),
                ticket_id,
            ),
            label_visibility="collapsed",
        )

        summary_col, detail_col = st.columns([1, 2], gap="large")

        with summary_col:
            for match in matches:
                score = float(match["score"])
                if match["ticket_id"] == selected_match_id:
                    st.markdown(f"**{match['ticket_id']}**")
                else:
                    st.markdown(match["ticket_id"])
                if score >= min_score:
                    st.success(f"score={score:.4f}")
                else:
                    st.warning(f"score={score:.4f}")
                st.caption(f"{match.get('issue_category', 'unknown')} | {match.get('title', '(untitled)')}")
                st.write(match.get("query") or "")
                st.markdown("---")

        with detail_col:
            selected_match = next(match for match in matches if match["ticket_id"] == selected_match_id)
            st.subheader(f"Retrieved Ticket: {selected_match['ticket_id']}")
            st.caption(
                f"Score: {selected_match['score']} | "
                f"Category: {selected_match.get('issue_category', 'unknown')} | "
                f"Title: {selected_match.get('title', '(untitled)')}"
            )

            st.markdown("**Indexed Issue Query**")
            st.write(selected_match.get("query") or "")

            st.markdown("**Indexed Resolution**")
            st.info(selected_match.get("positive") or "")

            historical_ticket = history_map.get(str(selected_match["ticket_id"]))
            if historical_ticket:
                st.markdown("**Historical Full Thread**")
                render_ticket_thread(historical_ticket)
            else:
                st.caption("Provide a historical threads JSONL file to view the full retrieved conversation.")


if __name__ == "__main__":
    main()

