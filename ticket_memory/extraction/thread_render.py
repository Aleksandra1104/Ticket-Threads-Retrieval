from __future__ import annotations

from typing import List

MAX_THREAD_CHARS = 14000


def clean_training_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split()).strip()


def role_tag(role: str) -> str:
    if role == "agent":
        return "AGENT"
    if role == "requester":
        return "REQUESTER"
    return "UNKNOWN"


def build_thread_text(ticket: object) -> str:
    lines: List[str] = []
    title = clean_training_text(getattr(ticket, "title", ""))
    description = clean_training_text(getattr(ticket, "description", ""))
    messages = getattr(ticket, "messages", [])

    if title:
        lines.append(f"Ticket Title: {title}")
    if description:
        lines.append(f"Ticket Description: {description}")
    lines.append("Thread:")

    for idx, msg in enumerate(messages):
        msg_text = clean_training_text(getattr(msg, "text", ""))
        if not msg_text:
            continue
        header = f"[{idx}] {role_tag(getattr(msg, 'author_role', 'unknown'))}"
        author_name = getattr(msg, "author_name", "")
        created_at = getattr(msg, "created_at", "")
        if author_name:
            header += f" ({author_name})"
        if created_at:
            header += f" @ {created_at}"
        lines.append(header)
        lines.append(msg_text)

    chunks: List[str] = []
    current_length = 0
    for line in lines:
        extra = len(line) + (1 if chunks else 0)
        if chunks and current_length + extra > MAX_THREAD_CHARS:
            break
        chunks.append(line)
        current_length += extra
    return "\n".join(chunks).strip()

