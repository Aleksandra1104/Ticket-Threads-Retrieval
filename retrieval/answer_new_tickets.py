#!/usr/bin/env python3
"""
Answer new ticket JSON/JSONL by retrieving similar resolved tickets.

This script:
1. Loads a local embedding index built from resolved ticket pairs.
2. Converts new tickets into retrieval queries.
3. Finds the top-k similar historical resolutions.
4. Returns a suggested answer, optionally refined with Ollama.

Example
-------
python answer_new_tickets.py ^
  --input new_tickets.jsonl ^
  --model models/ticket-pairs ^
  --index-dir retrieval_index ^
  --output answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from extraction.extract_ticket_pairs import Ticket, clean_message_text as clean_training_text, load_tickets
from ticket_memory.extraction.thread_render import build_thread_text


DEFAULT_TOP_K = 3
DEFAULT_MIN_SCORE = 0.55
DEFAULT_TIMEOUT_SECONDS = 180
MAX_REQUESTER_CHARS = 1200

SYSTEM_PROMPT = """
You write concise IT support draft replies grounded only in retrieved historical ticket resolutions.

Rules:
- Use only the retrieved evidence.
- If the retrieved evidence is weak or ambiguous, say so plainly.
- Prefer actionable language.
- Do not invent environment-specific steps that were not present in the evidence.
- Keep the reply brief and suitable for a help desk response draft.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieve answers for new tickets using a local SentenceTransformer index.")
    parser.add_argument("--input", required=True, help="Path to new ticket JSON/JSONL/CSV")
    parser.add_argument("--model", required=True, help="SentenceTransformer model path or model name")
    parser.add_argument("--index-dir", required=True, help="Directory created by build_ticket_index.py")
    parser.add_argument("--output", required=True, help="Path to output JSONL")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of matches to return")
    parser.add_argument("--min-score", type=float, default=DEFAULT_MIN_SCORE, help="Minimum cosine similarity for confident suggestions")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--ollama-model", help="Optional Ollama model for grounded answer synthesis")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at line {line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_retrieval_query(ticket: Ticket) -> str:
    requester_messages: List[str] = []
    for message in ticket.messages:
        if message.author_role == "requester":
            text = clean_training_text(message.text)
            if text:
                requester_messages.append(text)

    parts: List[str] = []
    title = clean_training_text(ticket.title)
    description = clean_training_text(ticket.description)
    if title:
        parts.append(f"Title: {title}")
    if description:
        parts.append(f"Description: {description}")
    if requester_messages:
        requester_blob = " ".join(requester_messages)
        parts.append(f"Requester: {requester_blob[:MAX_REQUESTER_CHARS].strip()}")
    if not parts:
        parts.append(build_thread_text(ticket))
    return "\n".join(part for part in parts if part).strip()


def load_index(index_dir: Path) -> tuple[list[Dict[str, Any]], np.ndarray]:
    records_path = index_dir / "records.jsonl"
    embeddings_path = index_dir / "embeddings.npy"
    if not records_path.exists():
        raise ValueError(f"Index records file not found: {records_path}")
    if not embeddings_path.exists():
        raise ValueError(f"Index embeddings file not found: {embeddings_path}")

    records = read_jsonl(records_path)
    embeddings = np.load(embeddings_path)
    if len(records) != int(embeddings.shape[0]):
        raise ValueError("Index records and embeddings are out of sync.")
    return records, embeddings


def retrieve_matches(
    query_embedding: np.ndarray,
    records: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int,
) -> List[Dict[str, Any]]:
    scores = embeddings @ query_embedding
    if scores.ndim != 1:
        scores = scores.reshape(-1)

    top_indices = np.argsort(scores)[::-1][:top_k]
    matches: List[Dict[str, Any]] = []
    for index in top_indices:
        record = records[int(index)]
        matches.append(
            {
                "ticket_id": record.get("ticket_id"),
                "score": round(float(scores[int(index)]), 4),
                "query": record.get("query"),
                "positive": record.get("positive"),
                "issue_category": record.get("issue_category"),
                "title": record.get("title"),
            }
        )
    return matches


def draft_answer(matches: Sequence[Dict[str, Any]], min_score: float) -> tuple[str, str]:
    if not matches:
        return "No similar resolved tickets were found.", "no_match"

    best = matches[0]
    best_score = float(best["score"])
    best_resolution = str(best.get("positive") or "").strip()
    if best_score < min_score:
        return (
            f"No high-confidence match was found. Closest known resolution: {best_resolution}",
            "low_confidence",
        )
    return best_resolution, "retrieved_resolution"


def ollama_answer(
    base_url: str,
    model: str,
    timeout_seconds: int,
    ticket: Ticket,
    retrieval_query: str,
    matches: Sequence[Dict[str, Any]],
    min_score: float,
) -> str:
    evidence_lines: List[str] = []
    for idx, match in enumerate(matches, start=1):
        evidence_lines.append(
            f"{idx}. score={match['score']}, category={match.get('issue_category')}, "
            f"historical_issue={match.get('query')}, historical_resolution={match.get('positive')}"
        )

    user_prompt = (
        f"New ticket:\n{build_thread_text(ticket)}\n\n"
        f"Retrieval query:\n{retrieval_query}\n\n"
        f"Confidence threshold: {min_score}\n\n"
        f"Retrieved evidence:\n" + "\n".join(evidence_lines)
    )

    response = requests.post(
        f"{base_url.rstrip('/')}/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2},
            "keep_alive": 0,
        },
        timeout=timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    content = (((payload.get("message") or {}).get("content")) or "").strip()
    if not content:
        raise ValueError("Ollama returned empty content")
    return content


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    index_dir = Path(args.index_dir)

    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 1
    if args.top_k < 1:
        print("--top-k must be at least 1", file=sys.stderr)
        return 1

    try:
        records, embeddings = load_index(index_dir)
        tickets = list(load_tickets(input_path))
    except Exception as exc:  # noqa: BLE001
        print(f"Failed to load input data: {exc}", file=sys.stderr)
        return 1

    if not tickets:
        print("No tickets loaded.", file=sys.stderr)
        return 1

    model = SentenceTransformer(args.model)
    queries = [build_retrieval_query(ticket) for ticket in tickets]
    query_embeddings = model.encode(
        queries,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    rows: List[Dict[str, Any]] = []
    for ticket, retrieval_query, query_embedding in zip(tickets, queries, query_embeddings):
        matches = retrieve_matches(
            query_embedding=query_embedding,
            records=records,
            embeddings=embeddings,
            top_k=args.top_k,
        )
        suggested_answer, answer_source = draft_answer(matches, args.min_score)

        row: Dict[str, Any] = {
            "ticket_id": ticket.ticket_id,
            "status": ticket.status,
            "title": ticket.title,
            "retrieval_query": retrieval_query,
            "answer_source": answer_source,
            "suggested_answer": suggested_answer,
            "matches": matches,
        }

        if args.ollama_model:
            try:
                row["llm_answer"] = ollama_answer(
                    base_url=args.ollama_base_url,
                    model=args.ollama_model,
                    timeout_seconds=args.timeout_seconds,
                    ticket=ticket,
                    retrieval_query=retrieval_query,
                    matches=matches,
                    min_score=args.min_score,
                )
            except Exception as exc:  # noqa: BLE001
                row["llm_answer_error"] = str(exc)

        rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    print(f"Processed tickets: {len(rows)}")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

