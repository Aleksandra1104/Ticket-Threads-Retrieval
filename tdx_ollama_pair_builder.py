#!/usr/bin/env python3
"""
Build SentenceTransformer training pairs from resolved TeamsDynamix tickets
by using Ollama to analyze each ticket thread and extract:
  - the user's issue/query
  - the final resolution
  - a confidence score

Input
-----
Supports JSON, JSONL, or CSV exports.
Each ticket should contain enough information to reconstruct a thread.

Recommended normalized input shape per ticket (JSON / JSONL):
{
  "ticket_id": "12345",
  "status": "Closed",
  "title": "Cannot log in",
  "description": "I can't access my account",
  "messages": [
    {
      "author_name": "Jane User",
      "author_role": "requester",   # requester | agent | unknown
      "created_at": "2026-03-20T13:00:00Z",
      "text": "I can't log into my account"
    },
    {
      "author_name": "Help Desk",
      "author_role": "agent",
      "created_at": "2026-03-20T13:15:00Z",
      "text": "Your account was locked. I unlocked it. Please try again."
    }
  ]
}

CSV mode is supported too, but because thread exports differ by tenant, the script expects
these columns by default:
  - ticket_id
  - status
  - title
  - description
  - thread_json   (JSON array of messages)

You can adapt the `row_to_ticket_from_csv()` function for your export.

Outputs
-------
1) extracted_pairs.jsonl   # one structured record per ticket
2) train_pairs.csv        # sentence1, sentence2, label
3) triplets.jsonl         # anchor, positive, negative
4) skipped_tickets.jsonl  # tickets that could not be extracted cleanly

Example
-------
python tdx_ollama_pair_builder.py \
  --input tickets.jsonl \
  --output-dir out_pairs \
  --ollama-model qwen2.5:7b-instruct \
  --min-confidence 0.65 \
  --require-closed

Ollama
------
The script calls Ollama's local /api/chat endpoint with format='json'.
Make sure Ollama is running and the model is pulled first, for example:
  ollama pull qwen2.5:7b-instruct
  ollama serve

Notes
-----
- This script is conservative. It prefers skipping noisy tickets over generating bad pairs.
- Tune `MAX_THREAD_CHARS`, role mapping, and prompt as needed for your tenant.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import parser
import json
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import requests


# ------------------------------
# Configuration defaults
# ------------------------------
MAX_THREAD_CHARS = 14000
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_NEGATIVES_PER_POSITIVE = 1
RANDOM_SEED = 42
ALLOWED_ISSUE_CATEGORIES = {
    "account_locked",
    "password_reset",
    "vpn_issue",
    "email_issue",
    "printer_issue",
    "permission_issue",
    # kept for future expansion / real-world data
    "network_issue",
    "software_access",
    "hardware_issue",
    "phishing_report",
    "other",
}


# ------------------------------
# Data classes
# ------------------------------
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


@dataclass
class ExtractedPair:
    ticket_id: str
    query: str
    positive: str
    issue_category: str
    confidence: float
    reasoning_short: str
    used_message_indexes: List[int]
    title: str
    status: str
    metadata: Dict[str, Any]


# ------------------------------
# Helpers: text cleaning
# ------------------------------
SIGNOFF_PATTERNS = [
    r"\bthanks[,! ]*$",
    r"\bthank you[,! ]*$",
    r"\bbest regards[,! ]*$",
    r"\bregards[,! ]*$",
    r"\bsincerely[,! ]*$",
]


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def strip_email_headers(text: str) -> str:
    lines = text.splitlines()
    filtered: List[str] = []
    header_prefixes = (
        "from:",
        "to:",
        "cc:",
        "bcc:",
        "sent:",
        "subject:",
    )
    for line in lines:
        if line.strip().lower().startswith(header_prefixes):
            continue
        filtered.append(line)
    return "\n".join(filtered)


def strip_urls(text: str) -> str:
    return re.sub(r"https?://\S+", " ", text)


def strip_reply_markers(text: str) -> str:
    patterns = [
        r"^>.*$",
        r"^-{2,}\s*original message\s*-{2,}.*$",
        r"^on .* wrote:.*$",
    ]
    for pattern in patterns:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.MULTILINE)
    return text


def strip_signature_like_tail(text: str) -> str:
    lines = [ln.rstrip() for ln in text.splitlines()]
    cleaned: List[str] = []
    for line in lines:
        cleaned.append(line)
    text = "\n".join(cleaned).strip()
    for pattern in SIGNOFF_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


def clean_message_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text)
    text = strip_email_headers(text)
    text = strip_reply_markers(text)
    text = strip_urls(text)
    text = normalize_whitespace(text)
    text = strip_signature_like_tail(text)
    return text.strip()


def clean_training_text(text: str) -> str:
    text = clean_message_text(text)
    text = re.sub(r"\b(ticket|case|incident)\s*#?\d+\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ------------------------------
# Helpers: role mapping / ticket parsing
# ------------------------------
REQUESTER_ROLE_ALIASES = {
    "requester", "user", "customer", "client", "employee", "student", "faculty", "reporter"
}
AGENT_ROLE_ALIASES = {
    "agent", "technician", "tech", "it", "assignee", "staff", "service_desk", "analyst", "resolver"
}


def normalize_role(raw_role: Any) -> str:
    value = str(raw_role or "").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    if value in REQUESTER_ROLE_ALIASES:
        return "requester"
    if value in AGENT_ROLE_ALIASES:
        return "agent"
    return "unknown"


def parse_message(obj: Dict[str, Any]) -> Message:
    author_name = str(
        obj.get("author_name")
        or obj.get("author")
        or obj.get("created_by_name")
        or obj.get("name")
        or "unknown"
    )
    author_role = normalize_role(
        obj.get("author_role")
        or obj.get("role")
        or obj.get("created_by_role")
        or obj.get("sender_role")
    )
    created_at = str(obj.get("created_at") or obj.get("date") or obj.get("timestamp") or "")
    text = clean_message_text(
        obj.get("text")
        or obj.get("body")
        or obj.get("message")
        or obj.get("comment")
        or obj.get("content")
        or ""
    )
    return Message(author_name=author_name, author_role=author_role, created_at=created_at, text=text)


def row_to_ticket_from_csv(row: Dict[str, Any]) -> Ticket:
    ticket_id = str(row.get("ticket_id") or row.get("id") or row.get("Ticket ID") or "")
    status = str(row.get("status") or row.get("Status") or "")
    title = str(row.get("title") or row.get("Title") or "")
    description = clean_message_text(row.get("description") or row.get("Description") or "")

    thread_payload = row.get("thread_json") or row.get("messages") or row.get("ThreadJSON") or "[]"
    try:
        thread_items = json.loads(thread_payload) if isinstance(thread_payload, str) else thread_payload
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse thread_json for ticket {ticket_id}: {exc}") from exc

    messages = [parse_message(item) for item in thread_items if isinstance(item, dict)]
    return Ticket(
        ticket_id=ticket_id,
        status=status,
        title=title,
        description=description,
        messages=messages,
        raw=dict(row),
    )


def dict_to_ticket(obj: Dict[str, Any]) -> Ticket:
    ticket_id = str(obj.get("ticket_id") or obj.get("id") or obj.get("ticketId") or "")
    status = str(obj.get("status") or obj.get("Status") or "")
    title = str(obj.get("title") or obj.get("subject") or obj.get("Title") or "")
    description = clean_message_text(obj.get("description") or obj.get("body") or obj.get("Description") or "")

    raw_messages = obj.get("messages") or obj.get("thread") or obj.get("comments") or []
    if isinstance(raw_messages, str):
        try:
            raw_messages = json.loads(raw_messages)
        except json.JSONDecodeError:
            raw_messages = []

    messages = [parse_message(item) for item in raw_messages if isinstance(item, dict)]

    if description and not messages:
        messages = [
            Message(
                author_name="requester",
                author_role="requester",
                created_at="",
                text=description,
            )
        ]

    return Ticket(
        ticket_id=ticket_id,
        status=status,
        title=title,
        description=description,
        messages=messages,
        raw=obj,
    )


# ------------------------------
# Input loaders
# ------------------------------
def load_jsonl(path: Path) -> Iterator[Ticket]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("Expected JSON object per line")
                yield dict_to_ticket(obj)
            except Exception as exc:
                raise ValueError(f"Error parsing JSONL at line {line_no}: {exc}") from exc


def load_json(path: Path) -> Iterator[Ticket]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        items = data.get("tickets") or data.get("items") or [data]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("JSON input must be an object or a list")
    for idx, obj in enumerate(items):
        if not isinstance(obj, dict):
            raise ValueError(f"JSON item at index {idx} is not an object")
        yield dict_to_ticket(obj)


def load_csv(path: Path) -> Iterator[Ticket]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row_to_ticket_from_csv(row)


def load_tickets(path: Path) -> Iterator[Ticket]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return load_jsonl(path)
    if suffix == ".json":
        return load_json(path)
    if suffix == ".csv":
        return load_csv(path)
    raise ValueError("Unsupported input format. Use .jsonl, .json, or .csv")


# ------------------------------
# Ground-truth helpers
# ------------------------------
def get_ground_truth(ticket: Ticket) -> Dict[str, Any]:
    gt = ticket.raw.get("ground_truth")
    return gt if isinstance(gt, dict) else {}


def get_resolution_state(ticket: Ticket) -> str:
    gt = get_ground_truth(ticket)
    state = str(gt.get("resolution_state") or "").strip().lower()
    return state if state in {"resolved", "partial", "unresolved"} else ""


def has_secondary_issue(ticket: Ticket) -> bool:
    gt = get_ground_truth(ticket)
    return gt.get("secondary_issue_category") is not None


# ------------------------------
# Thread rendering for Ollama
# ------------------------------
def role_tag(role: str) -> str:
    if role == "agent":
        return "AGENT"
    if role == "requester":
        return "REQUESTER"
    return "UNKNOWN"


def build_thread_text(ticket: Ticket) -> str:
    lines: List[str] = []
    if ticket.title:
        lines.append(f"Ticket Title: {clean_training_text(ticket.title)}")
    if ticket.description:
        lines.append(f"Ticket Description: {clean_training_text(ticket.description)}")
    lines.append("Thread:")

    for idx, msg in enumerate(ticket.messages):
        msg_text = clean_training_text(msg.text)
        if not msg_text:
            continue
        header = f"[{idx}] {role_tag(msg.author_role)}"
        if msg.author_name:
            header += f" ({msg.author_name})"
        if msg.created_at:
            header += f" @ {msg.created_at}"
        lines.append(header)
        lines.append(msg_text)

    if not lines:
        return ""

    chunks: List[str] = []
    current_length = 0
    for line in lines:
        extra = len(line) + (1 if chunks else 0)
        if chunks and current_length + extra > MAX_THREAD_CHARS:
            break
        chunks.append(line)
        current_length += extra

    return "\n".join(chunks).strip()


# ------------------------------
# Ollama extraction
# ------------------------------
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "should_use": {"type": "boolean"},
        "issue_summary": {"type": "string"},
        "resolution_summary": {"type": "string"},
        "issue_category": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning_short": {"type": "string"},
        "used_message_indexes": {
            "type": "array",
            "items": {"type": "integer"}
        }
    },
    "required": [
        "should_use",
        "issue_summary",
        "resolution_summary",
        "issue_category",
        "confidence",
        "reasoning_short",
        "used_message_indexes",
    ]
}


SYSTEM_PROMPT = """
You extract high-quality training pairs for an IT support retrieval system.

Your task:
Given one support ticket thread, extract:
- issue_summary: a short natural-language statement of the user's actual problem
- resolution_summary: one sentence containing BOTH:
  (a) the underlying cause, if stated or strongly implied in the thread
  (b) the fix that resolved it
- issue_category: choose EXACTLY ONE from this closed list:
  account_locked
  password_reset
  vpn_issue
  email_issue
  printer_issue
  permission_issue
  network_issue
  software_access
  hardware_issue
  phishing_report
  other

Important rules:
1. Use only information supported by the thread.
2. Do not invent causes that are not stated or strongly implied.
3. Prefer the user's real problem, not greetings, pleasantries, or follow-up confirmations.
4. Prefer the final successful fix, not intermediate troubleshooting attempts.
5. If the thread does not clearly contain a real resolution, set should_use=false.
6. If the issue is about Outlook, mailbox loading, email access, repeated credential prompts for email, or desktop email client authentication, classify it as email_issue, NOT password_reset.
7. If the issue is about reset links, expired reset tokens, forgot password flows, or password reset emails, classify it as password_reset.
8. issue_summary should be concise and natural, usually 4 to 12 words.
9. resolution_summary should be one sentence, concise but specific, and should follow this style:
   "Cause: <cause>. Fix: <fix/result>."
10. Do not use vague summaries like:
   - "issue fixed"
   - "resolved by IT"
   - "user can log in now"
11. confidence must be realistic:
   - 0.9 to 1.0 only if the issue and fix are both explicit
   - 0.7 to 0.89 if mostly clear with small ambiguity
   - 0.5 to 0.69 if somewhat inferred
   - below 0.5 if unclear, and usually set should_use=false
12. reasoning_short must be one brief sentence explaining why the pair is valid.
13. used_message_indexes must include the most relevant supporting messages.
14. User text may contain typos, informal wording, and messy phrasing; interpret the intended meaning.
15. If the thread contains multiple issues, choose the single primary issue that is most clearly resolved.
16. If a secondary issue appears but is not resolved, do not use it as the basis for resolution_summary.

Return only valid JSON matching the schema.
""".strip()


USER_PROMPT_TEMPLATE = """
Analyze this support ticket thread and return ONLY valid JSON matching the requested schema.

{thread_text}
""".strip()


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def extract_pair(self, thread_text: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": self.model,
            "stream": False,
            "format": EXTRACTION_SCHEMA,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(thread_text=thread_text)},
            ],
            "options": {
                "temperature": 0,
            },
            "keep_alive": 0,
        }

        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout_seconds)
                response.raise_for_status()

                data = response.json()
                content = (((data.get("message") or {}).get("content")) or "").strip()
                if not content:
                    raise ValueError("Ollama returned empty content")

                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Ollama returned non-JSON content: {content[:500]}") from exc

                return parsed

            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
                requests.RequestException,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc

                is_last_attempt = attempt == self.max_retries
                if is_last_attempt:
                    break

                sleep_for = self.retry_backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"Ollama call failed (attempt {attempt}/{self.max_retries}): {exc}. "
                    f"Retrying in {sleep_for:.1f}s..."
                )
                time.sleep(sleep_for)

        raise RuntimeError(
            f"Ollama extraction failed after {self.max_retries} attempts: {last_error}"
        ) from last_error


# ------------------------------
# Pair extraction pipeline
# ------------------------------
def is_closed_status(status: str) -> bool:
    s = str(status or "").strip().lower()
    return s in {"closed", "resolved", "complete", "completed", "done"}


def has_real_thread(ticket: Ticket) -> bool:
    meaningful_messages = [m for m in ticket.messages if clean_training_text(m.text)]
    return bool(ticket.title or ticket.description or meaningful_messages)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def validate_extraction(raw: Dict[str, Any], min_confidence: float) -> Tuple[bool, str]:
    should_use = bool(raw.get("should_use"))
    issue = clean_training_text(str(raw.get("issue_summary") or ""))
    resolution = clean_training_text(str(raw.get("resolution_summary") or ""))
    issue_category = clean_training_text(str(raw.get("issue_category") or "other")).lower()
    confidence = raw.get("confidence")

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return False, "invalid confidence"

    if not should_use:
        return False, "model marked ticket unusable"
    if confidence_value < min_confidence:
        return False, f"confidence below threshold ({confidence_value:.2f} < {min_confidence:.2f})"
    if len(issue) < 8:
        return False, "issue summary too short"
    if len(resolution) < 8:
        return False, "resolution summary too short"
    if issue_category not in ALLOWED_ISSUE_CATEGORIES:
        return False, f"invalid issue_category: {issue_category}"

    bad_resolution_fragments = [
        "let me know",
        "please advise",
        "please provide",
        "can you send",
        "thank you",
        "we are looking into it",
        "escalated",
        "following up",
        "not fully resolved",
        "partially resolved",
        "requires escalation",
        "still investigating",
    ]
    lower_res = resolution.lower()
    if any(fragment in lower_res for fragment in bad_resolution_fragments):
        return False, "resolution looks like ongoing conversation rather than final fix"

    return True, "ok"


def extract_pairs(
    tickets: Iterable[Ticket],
    ollama: OllamaClient,
    min_confidence: float,
    require_closed: bool,
    sleep_seconds: float,
    skip_mixed_issues: bool,
) -> Tuple[List[ExtractedPair], List[Dict[str, Any]], int]:
    extracted: List[ExtractedPair] = []
    skipped: List[Dict[str, Any]] = []
    loaded_count = 0

    for idx, ticket in enumerate(tickets, start=1):
        loaded_count = idx
        if idx % 5 == 0 or idx == 1:
            print(f"Processing ticket {idx}...")

        gt = get_ground_truth(ticket)
        resolution_state = get_resolution_state(ticket)

        if require_closed:
            if resolution_state:
                if resolution_state != "resolved":
                    skipped.append({
                        "ticket_id": ticket.ticket_id,
                        "reason": f"ground_truth resolution_state is not resolved: {resolution_state}",
                    })
                    continue
            elif not is_closed_status(ticket.status):
                skipped.append({
                    "ticket_id": ticket.ticket_id,
                    "reason": f"status not closed/resolved: {ticket.status}",
                })
                continue

        if skip_mixed_issues and has_secondary_issue(ticket):
            skipped.append({
                "ticket_id": ticket.ticket_id,
                "reason": "mixed-issue ticket skipped for single-label extraction",
            })
            continue

        if not has_real_thread(ticket):
            skipped.append({
                "ticket_id": ticket.ticket_id,
                "reason": "empty or unusable thread",
            })
            continue

        thread_text = build_thread_text(ticket)

        try:
            raw = ollama.extract_pair(thread_text)
            is_valid, reason = validate_extraction(raw, min_confidence=min_confidence)
            if not is_valid:
                skipped.append({
                    "ticket_id": ticket.ticket_id,
                    "reason": reason,
                    "ollama_output": raw,
                    "ground_truth_primary_issue_category": gt.get("primary_issue_category"),
                    "ground_truth_secondary_issue_category": gt.get("secondary_issue_category"),
                    "ground_truth_resolution_state": gt.get("resolution_state"),
                })
                continue

            pair = ExtractedPair(
                ticket_id=ticket.ticket_id or f"ticket_{idx}_{stable_hash(thread_text)}",
                query=clean_training_text(str(raw["issue_summary"])),
                positive=clean_training_text(str(raw["resolution_summary"])),
                issue_category=clean_training_text(str(raw.get("issue_category") or "other")).lower(),
                confidence=float(raw["confidence"]),
                reasoning_short=clean_training_text(str(raw.get("reasoning_short") or "")),
                used_message_indexes=[int(x) for x in raw.get("used_message_indexes", []) if isinstance(x, int)],
                title=clean_training_text(ticket.title),
                status=ticket.status,
                metadata={
                    "raw_ollama": raw,
                    "ground_truth_primary_issue_category": gt.get("primary_issue_category"),
                    "ground_truth_secondary_issue_category": gt.get("secondary_issue_category"),
                    "ground_truth_resolution_state": gt.get("resolution_state"),
                },
            )
            extracted.append(pair)
            if idx % 5 == 0 or idx == 1:
                print(f"  ✓ extracted (total good: {len(extracted)})")

        except Exception as exc:
            skipped.append({
                "ticket_id": ticket.ticket_id,
                "reason": f"extraction error: {exc}",
                "ground_truth_primary_issue_category": gt.get("primary_issue_category"),
                "ground_truth_secondary_issue_category": gt.get("secondary_issue_category"),
                "ground_truth_resolution_state": gt.get("resolution_state"),
            })
            if idx % 5 == 0 or idx == 1:
                print(f"  ✗ skipped (total skipped: {len(skipped)})")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return extracted, skipped, loaded_count


# ------------------------------
# Negative mining
# ------------------------------
def group_by_category(pairs: Sequence[ExtractedPair]) -> Dict[str, List[ExtractedPair]]:
    grouped: Dict[str, List[ExtractedPair]] = {}
    for pair in pairs:
        grouped.setdefault(pair.issue_category or "other", []).append(pair)
    return grouped


def choose_negative(
    anchor: ExtractedPair,
    all_pairs: Sequence[ExtractedPair],
    by_category: Dict[str, List[ExtractedPair]],
) -> Optional[str]:
    other_categories = [cat for cat in by_category.keys() if cat != anchor.issue_category]
    random.shuffle(other_categories)

    for cat in other_categories:
        candidates = by_category.get(cat, [])
        if candidates:
            choice = random.choice(candidates)
            if choice.positive != anchor.positive:
                return choice.positive

    fallback = [p for p in all_pairs if p.ticket_id != anchor.ticket_id and p.positive != anchor.positive]
    if fallback:
        return random.choice(fallback).positive
    return None


# ------------------------------
# Writers
# ------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_extracted_pairs(path: Path, pairs: Sequence[ExtractedPair]) -> None:
    rows = []
    for p in pairs:
        rows.append({
            "ticket_id": p.ticket_id,
            "query": p.query,
            "positive": p.positive,
            "issue_category": p.issue_category,
            "confidence": p.confidence,
            "reasoning_short": p.reasoning_short,
            "used_message_indexes": p.used_message_indexes,
            "title": p.title,
            "status": p.status,
            "ground_truth_primary_issue_category": p.metadata.get("ground_truth_primary_issue_category"),
            "ground_truth_secondary_issue_category": p.metadata.get("ground_truth_secondary_issue_category"),
            "ground_truth_resolution_state": p.metadata.get("ground_truth_resolution_state"),
            "metadata": p.metadata,
        })
    write_jsonl(path, rows)


def write_train_pairs_csv(
    path: Path,
    pairs: Sequence[ExtractedPair],
    negatives_per_positive: int,
) -> None:
    random.seed(RANDOM_SEED)
    by_category = group_by_category(pairs)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
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
            writer.writerow({
                "sentence1": pair.query,
                "sentence2": pair.positive,
                "label": 1.0,
                "ticket_id": pair.ticket_id,
                "issue_category": pair.issue_category,
                "ground_truth_primary_issue_category": pair.metadata.get("ground_truth_primary_issue_category"),
                "ground_truth_resolution_state": pair.metadata.get("ground_truth_resolution_state"),
            })

            for _ in range(negatives_per_positive):
                negative = choose_negative(pair, pairs, by_category)
                if not negative:
                    continue
                writer.writerow({
                    "sentence1": pair.query,
                    "sentence2": negative,
                    "label": 0.0,
                    "ticket_id": pair.ticket_id,
                    "issue_category": pair.issue_category,
                    "ground_truth_primary_issue_category": pair.metadata.get("ground_truth_primary_issue_category"),
                    "ground_truth_resolution_state": pair.metadata.get("ground_truth_resolution_state"),
                })


def write_triplets_jsonl(path: Path, pairs: Sequence[ExtractedPair]) -> None:
    random.seed(RANDOM_SEED)
    by_category = group_by_category(pairs)
    rows: List[Dict[str, Any]] = []

    for pair in pairs:
        negative = choose_negative(pair, pairs, by_category)
        if not negative:
            continue
        rows.append({
            "anchor": pair.query,
            "positive": pair.positive,
            "negative": negative,
            "ticket_id": pair.ticket_id,
            "issue_category": pair.issue_category,
            "ground_truth_primary_issue_category": pair.metadata.get("ground_truth_primary_issue_category"),
            "ground_truth_resolution_state": pair.metadata.get("ground_truth_resolution_state"),
        })

    write_jsonl(path, rows)


# ------------------------------
# Main
# ------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build training pairs from TeamsDynamix ticket threads using Ollama.")
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

    ollama = OllamaClient(
        base_url=args.ollama_base_url,
        model=args.ollama_model,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.ollama_max_retries,
        retry_backoff_seconds=args.ollama_retry_backoff,
    )

    extracted_pairs, skipped, loaded_count = extract_pairs(
        tickets=tickets,
        ollama=ollama,
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
    write_train_pairs_csv(
        train_pairs_path,
        extracted_pairs,
        negatives_per_positive=max(0, args.negatives_per_positive),
    )
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


