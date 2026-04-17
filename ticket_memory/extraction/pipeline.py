from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .models import ExtractedPairRecord
from .thread_render import build_thread_text, clean_training_text

ALLOWED_ISSUE_CATEGORIES = {
    "account_locked",
    "password_reset",
    "mfa_issue",
    "permission_issue",
    "onboarding_offboarding",

    "vpn_issue",
    "wifi_connectivity",
    "internet_access",
    "voip_telephony",

    "workstation_failure",
    "peripheral_issue",
    "printer_issue",
    "mobile_device_issue",

    "email_issue",
    "software_install",
    "application_crash",
    "browser_issue",

    "shared_drive_issue",
    "data_recovery",
    "disk_space_full",

    "phishing_report",
    "malware_infection",
    "encryption_issue",

    "server_unavailable",
    "database_connection",
    "api_failure",

    "other",
}


def get_ground_truth(ticket: object) -> Dict[str, Any]:
    raw = getattr(ticket, "raw", {})
    gt = raw.get("ground_truth") if isinstance(raw, dict) else None
    return gt if isinstance(gt, dict) else {}


def get_resolution_state(ticket: object) -> str:
    gt = get_ground_truth(ticket)
    state = str(gt.get("resolution_state") or "").strip().lower()
    return state if state in {"resolved", "partial", "unresolved"} else ""


def has_secondary_issue(ticket: object) -> bool:
    gt = get_ground_truth(ticket)
    return gt.get("secondary_issue_category") is not None


def is_closed_status(status: str) -> bool:
    return str(status or "").strip().lower() in {"closed", "resolved", "complete", "completed", "done"}


def has_real_thread(ticket: object) -> bool:
    meaningful_messages = [m for m in getattr(ticket, "messages", []) if clean_training_text(getattr(m, "text", ""))]
    return bool(getattr(ticket, "title", "") or getattr(ticket, "description", "") or meaningful_messages)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def validate_extraction(raw: Dict[str, Any], min_confidence: float) -> Tuple[bool, str]:
    should_use = bool(raw.get("should_use"))
    issue = clean_training_text(str(raw.get("issue_summary") or ""))
    resolution = clean_training_text(str(raw.get("resolution_summary") or ""))
    issue_category = clean_training_text(str(raw.get("issue_category") or "other")).lower()

    try:
        confidence_value = float(raw.get("confidence"))
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

    lower_res = resolution.lower()
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
    if any(fragment in lower_res for fragment in bad_resolution_fragments):
        return False, "resolution looks like ongoing conversation rather than final fix"
    return True, "ok"


def extract_ticket_pairs(
    tickets: Iterable[object],
    extractor: object,
    min_confidence: float,
    require_closed: bool,
    sleep_seconds: float,
    skip_mixed_issues: bool,
) -> Tuple[List[ExtractedPairRecord], List[Dict[str, Any]], int]:
    extracted: List[ExtractedPairRecord] = []
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
                    skipped.append({"ticket_id": getattr(ticket, "ticket_id", ""), "reason": f"ground_truth resolution_state is not resolved: {resolution_state}"})
                    continue
            elif not is_closed_status(getattr(ticket, "status", "")):
                skipped.append({"ticket_id": getattr(ticket, "ticket_id", ""), "reason": f"status not closed/resolved: {getattr(ticket, 'status', '')}"})
                continue

        if skip_mixed_issues and has_secondary_issue(ticket):
            skipped.append({"ticket_id": getattr(ticket, "ticket_id", ""), "reason": "mixed-issue ticket skipped for single-label extraction"})
            continue

        if not has_real_thread(ticket):
            skipped.append({"ticket_id": getattr(ticket, "ticket_id", ""), "reason": "empty or unusable thread"})
            continue

        thread_text = build_thread_text(ticket)

        try:
            raw = extractor.extract_pair(thread_text)
            is_valid, reason = validate_extraction(raw, min_confidence=min_confidence)
            if not is_valid:
                skipped.append(
                    {
                        "ticket_id": getattr(ticket, "ticket_id", ""),
                        "reason": reason,
                        "ollama_output": raw,
                        # "ground_truth_primary_issue_category": gt.get("primary_issue_category"),
                        # "ground_truth_secondary_issue_category": gt.get("secondary_issue_category"),
                        # "ground_truth_resolution_state": gt.get("resolution_state"),
                    }
                )
                continue

            extracted.append(
                ExtractedPairRecord(
                    ticket_id=getattr(ticket, "ticket_id", "") or f"ticket_{idx}_{stable_hash(thread_text)}",
                    query=clean_training_text(str(raw["issue_summary"])),
                    positive=clean_training_text(str(raw["resolution_summary"])),
                    issue_category=clean_training_text(str(raw.get("issue_category") or "other")).lower(),
                    confidence=float(raw["confidence"]),
                    reasoning_short=clean_training_text(str(raw.get("reasoning_short") or "")),
                    used_message_indexes=[int(x) for x in raw.get("used_message_indexes", []) if isinstance(x, int)],
                    title=clean_training_text(getattr(ticket, "title", "")),
                    status=getattr(ticket, "status", ""),
                    metadata={
                        "raw_ollama": raw,
                        # "ground_truth_primary_issue_category": gt.get("primary_issue_category"),
                        # "ground_truth_secondary_issue_category": gt.get("secondary_issue_category"),
                        # "ground_truth_issue_titles": gt.get("issue_titles"),
                        # "ground_truth_resolution_summary": gt.get("resolution_summary"),
                        # "ground_truth_resolution_state": gt.get("resolution_state"),
                    },
                )
            )
            if idx % 5 == 0 or idx == 1:
                print(f"  extracted (total good: {len(extracted)})")
        except Exception as exc:
            skipped.append(
                {
                    "ticket_id": getattr(ticket, "ticket_id", ""),
                    "reason": f"extraction error: {exc}",
                    # "ground_truth_primary_issue_category": gt.get("primary_issue_category"),
                    # "ground_truth_secondary_issue_category": gt.get("secondary_issue_category"),
                    # "ground_truth_resolution_state": gt.get("resolution_state"),
                }
            )
            if idx % 5 == 0 or idx == 1:
                print(f"  skipped (total skipped: {len(skipped)})")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    return extracted, skipped, loaded_count

