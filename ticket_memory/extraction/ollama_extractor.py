from __future__ import annotations

import json
import time
from typing import Any, Dict

import requests


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
            "items": {"type": "integer"},
        },
    },
    "required": [
        "should_use",
        "issue_summary",
        "resolution_summary",
        "issue_category",
        "confidence",
        "reasoning_short",
        "used_message_indexes",
    ],
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
    mfa_issue
    permission_issue
    onboarding_offboarding

    vpn_issue
    wifi_connectivity
    internet_access
    voip_telephony

    workstation_failure
    peripheral_issue
    printer_issue
    mobile_device_issue

    email_issue
    software_install
    application_crash
    browser_issue

    shared_drive_issue
    data_recovery
    disk_space_full

    phishing_report
    malware_infection
    encryption_issue

    server_unavailable
    database_connection
    api_failure

    other

Important rules:
1. Use only information supported by the thread.
2. Do not invent causes that are not stated or strongly implied.
3. Prefer the user's real problem, not greetings, pleasantries, or follow-up confirmations.
4. Prefer the final successful fix, not intermediate troubleshooting attempts.
5. If the thread does not clearly contain a real resolution, set should_use=false. Tickets involving security incidents, phishing reports, malware alerts,
or other confirmed incident-response workflows SHOULD still be marked usable if the thread contains a clear issue and remediation action.
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


class OllamaThreadPairExtractor:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout_seconds: int = 300,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.5,
    ) -> None:
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
            "options": {"temperature": 0},
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
                return json.loads(content)
            except (
                requests.Timeout,
                requests.ConnectionError,
                requests.HTTPError,
                requests.RequestException,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                last_error = exc
                if attempt == self.max_retries:
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


