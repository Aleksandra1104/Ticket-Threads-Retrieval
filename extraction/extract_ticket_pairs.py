#!/usr/bin/env python3
"""
Clean CLI entrypoint for extracting issue-resolution pairs from support ticket threads.

This uses the modular extraction layer under ticket_memory/extraction/ and writes
the same downstream artifacts as tdx_ollama_pair_builder.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ticket_memory.extraction.ollama_extractor import OllamaThreadPairExtractor
from ticket_memory.extraction.pipeline import extract_ticket_pairs

RANDOM_SEED = 42
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_NEGATIVES_PER_POSITIVE = 1


@dataclass
class Message:
    author_name: str
    author_role: str
    created_at: str
    text: str


@dataclass
class Ticket:
    ticket_id: str
    status: str
    title: str
    description: str
    messages: List[Message]
    raw: Dict[str, Any]


REQUESTER_ROLE_ALIASES = {
    "requester", "user", "customer", "client", "employee", "student", "faculty", "reporter"
}
AGENT_ROLE_ALIASES = {
    "agent", "technician", "tech", "it", "assignee", "staff", "service_desk", "analyst", "resolver"
}


def normalize_role(raw_role: Any) -> str:
    value = str(raw_role or "").strip().lower().replace("-", "_").replace(" ", "_")
    if value in REQUESTER_ROLE_ALIASES:
        return "requester"
    if value in AGENT_ROLE_ALIASES:
        return "agent"
    return "unknown"


def clean_message_text(text: Any) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split()).strip()


def parse_message(obj: Dict[str, Any]) -> Message:
    return Message(
        author_name=str(obj.get("author_name") or obj.get("author") or obj.get("created_by_name") or obj.get("name") or "unknown"),
        author_role=normalize_role(obj.get("author_role") or obj.get("role") or obj.get("authorType")),
        created_at=str(obj.get("created_at") or obj.get("created") or obj.get("timestamp") or ""),
        text=clean_message_text(obj.get("text") or obj.get("body") or obj.get("message") or obj.get("content") or ""),
    )


def dict_to_ticket(obj: Dict[str, Any]) -> Ticket:
    raw_messages = obj.get("messages") or obj.get("thread") or []
    messages = [parse_message(item) for item in raw_messages if isinstance(item, dict)]
    return Ticket(
        ticket_id=str(obj.get("ticket_id") or obj.get("id") or ""),
        status=str(obj.get("status") or ""),
        title=clean_message_text(obj.get("title") or ""),
        description=clean_message_text(obj.get("description") or ""),
        messages=messages,
        raw=obj,
    )


def row_to_ticket_from_csv(row: Dict[str, Any]) -> Ticket:
    thread_json = row.get("thread_json") or "[]"
    try:
        raw_messages = json.loads(thread_json)
    except json.JSONDecodeError:
        raw_messages = []
    return Ticket(
        ticket_id=str(row.get("ticket_id") or ""),
        status=str(row.get("status") or ""),
        title=clean_message_text(row.get("title") or ""),
        description=clean_message_text(row.get("description") or ""),
        messages=[parse_message(item) for item in raw_messages if isinstance(item, dict)],
        raw=dict(row),
    )


def load_jsonl(path: Path) -> Iterator[Ticket]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at line {line_no}")
            yield dict_to_ticket(obj)


def load_json(path: Path) -> Iterator[Ticket]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list):
        raise ValueError("Expected JSON object or list of objects")
    for obj in payload:
        if isinstance(obj, dict):
            yield dict_to_ticket(obj)


def load_csv(path: Path) -> Iterator[Ticket]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row_to_ticket_from_csv(row)


def load_tickets(path: Path) -> Iterator[Ticket]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from load_jsonl(path)
        return
    if suffix == ".json":
        yield from load_json(path)
        return
    if suffix == ".csv":
        yield from load_csv(path)
        return
    raise ValueError(f"Unsupported input format: {path.suffix}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_extracted_pairs(path: Path, pairs: Sequence[Any]) -> None:
    rows = []
    for pair in pairs:
        rows.append(
            {
                "ticket_id": pair.ticket_id,
                "query": pair.query,
                "positive": pair.positive,
                "issue_category": pair.issue_category,
                "confidence": pair.confidence,
                "reasoning_short": pair.reasoning_short,
                "used_message_indexes": pair.used_message_indexes,
                "title": pair.title,
                "status": pair.status,
                # "ground_truth_primary_issue_category": pair.metadata.get("ground_truth_primary_issue_category"),
                # "ground_truth_secondary_issue_category": pair.metadata.get("ground_truth_secondary_issue_category"),
                # "ground_truth_resolution_state": pair.metadata.get("ground_truth_resolution_state"),
                "metadata": pair.metadata,
            }
        )
    write_jsonl(path, rows)


def group_by_category(pairs: Sequence[Any]) -> Dict[str, List[Any]]:
    grouped: Dict[str, List[Any]] = {}
    for pair in pairs:
        grouped.setdefault(pair.issue_category or "other", []).append(pair)
    return grouped


def choose_negative(anchor: Any, all_pairs: Sequence[Any], by_category: Dict[str, List[Any]]) -> Optional[str]:
    other_categories = [category for category in by_category.keys() if category != anchor.issue_category]
    random.shuffle(other_categories)
    for category in other_categories:
        candidates = by_category.get(category, [])
        if candidates:
            choice = random.choice(candidates)
            if choice.positive != anchor.positive:
                return choice.positive
    fallback = [pair for pair in all_pairs if pair.ticket_id != anchor.ticket_id and pair.positive != anchor.positive]
    if fallback:
        return random.choice(fallback).positive
    return None


def write_train_pairs_csv(path: Path, pairs: Sequence[Any], negatives_per_positive: int) -> None:
    random.seed(RANDOM_SEED)
    by_category = group_by_category(pairs)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sentence1",
                "sentence2",
                "label",
                "ticket_id",
                "issue_category",
                "ground_truth_primary_issue_category",
                "ground_truth_resolution_state",
            ],
        )
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "sentence1": pair.query,
                    "sentence2": pair.positive,
                    "label": 1.0,
                    "ticket_id": pair.ticket_id,
                    "issue_category": pair.issue_category,
                    "ground_truth_primary_issue_category": pair.metadata.get("ground_truth_primary_issue_category"),
                    "ground_truth_resolution_state": pair.metadata.get("ground_truth_resolution_state"),
                }
            )
            for _ in range(negatives_per_positive):
                negative = choose_negative(pair, pairs, by_category)
                if not negative:
                    continue
                writer.writerow(
                    {
                        "sentence1": pair.query,
                        "sentence2": negative,
                        "label": 0.0,
                        "ticket_id": pair.ticket_id,
                        "issue_category": pair.issue_category,
                        "ground_truth_primary_issue_category": pair.metadata.get("ground_truth_primary_issue_category"),
                        "ground_truth_resolution_state": pair.metadata.get("ground_truth_resolution_state"),
                    }
                )


def write_triplets_jsonl(path: Path, pairs: Sequence[Any]) -> None:
    random.seed(RANDOM_SEED)
    by_category = group_by_category(pairs)
    rows: List[Dict[str, Any]] = []
    for pair in pairs:
        negative = choose_negative(pair, pairs, by_category)
        if not negative:
            continue
        rows.append(
            {
                "anchor": pair.query,
                "positive": pair.positive,
                "negative": negative,
                "ticket_id": pair.ticket_id,
                "issue_category": pair.issue_category,
                "ground_truth_primary_issue_category": pair.metadata.get("ground_truth_primary_issue_category"),
                "ground_truth_resolution_state": pair.metadata.get("ground_truth_resolution_state"),
            }
        )
    write_jsonl(path, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract issue-resolution pairs from support ticket threads using Ollama.")
    parser.add_argument("--input", required=True, help="Path to .jsonl, .json, or .csv ticket export")
    parser.add_argument("--output-dir", required=True, help="Directory for outputs")
    parser.add_argument("--ollama-model", default="qwen2.5:7b-instruct", help="Ollama model name")
    parser.add_argument("--ollama-base-url", default="http://localhost:11434", help="Ollama base URL")
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--min-confidence", type=float, default=0.65)
    parser.add_argument("--require-closed", action="store_true", help="Only use tickets with closed/resolved-like status")
    parser.add_argument("--skip-mixed-issues", action="store_true", help="Skip tickets that contain a secondary issue in ground_truth")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Pause between Ollama calls")
    parser.add_argument("--negatives-per-positive", type=int, default=DEFAULT_NEGATIVES_PER_POSITIVE)
    parser.add_argument("--ollama-max-retries", type=int, default=3, help="Number of retry attempts for Ollama calls")
    parser.add_argument("--ollama-retry-backoff", type=float, default=1.5, help="Base backoff in seconds between Ollama retries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(RANDOM_SEED)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    try:
        tickets = load_tickets(input_path)
    except ValueError as exc:
        print(f"Failed to load tickets: {exc}", file=sys.stderr)
        return 1

    extractor = OllamaThreadPairExtractor(
        base_url=args.ollama_base_url,
        model=args.ollama_model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.ollama_max_retries,
        retry_backoff_seconds=args.ollama_retry_backoff,
    )

    extracted_pairs, skipped, loaded_count = extract_ticket_pairs(
        tickets=tickets,
        extractor=extractor,
        min_confidence=args.min_confidence,
        require_closed=args.require_closed,
        sleep_seconds=args.sleep_seconds,
        skip_mixed_issues=args.skip_mixed_issues,
    )
    if loaded_count == 0:
        print("No tickets loaded.", file=sys.stderr)
        return 1

    extracted_path = output_dir / "extracted_pairs.jsonl"
    skipped_path = output_dir / "skipped_tickets.jsonl"
    train_pairs_path = output_dir / "train_pairs.csv"
    triplets_path = output_dir / "triplets.jsonl"

    write_extracted_pairs(extracted_path, extracted_pairs)
    write_jsonl(skipped_path, skipped)
    write_train_pairs_csv(train_pairs_path, extracted_pairs, negatives_per_positive=max(0, args.negatives_per_positive))
    write_triplets_jsonl(triplets_path, extracted_pairs)

    print(f"Loaded tickets:        {loaded_count}")
    print(f"Extracted good pairs:  {len(extracted_pairs)}")
    print(f"Skipped tickets:       {len(skipped)}")
    print(f"Wrote: {extracted_path}")
    print(f"Wrote: {train_pairs_path}")
    print(f"Wrote: {triplets_path}")
    print(f"Wrote: {skipped_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

