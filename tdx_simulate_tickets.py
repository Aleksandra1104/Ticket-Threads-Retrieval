#!/usr/bin/env python3
"""
Generate realistic synthetic TeamsDynamix-style ticket threads for testing
an extraction, classification, and summarization pipeline.

Output format: JSONL
Each line contains one ticket object with:
- ticket_id
- status
- title
- description
- messages
- ground_truth   (included only for evaluation; not present in real tickets)

The simulator is designed to create more realistic help desk data by generating:
- short and longer conversations, typically around 6 to 12+ messages per ticket
- clarification questions and additional follow-up exchanges
- failed first-step troubleshooting attempts
- split diagnoses and multi-step resolutions
- user confirmations and agent closure messages
- occasional unresolved tickets that should be skipped during extraction
- partially resolved tickets that improve but still require escalation
- mixed-issue tickets where a secondary problem appears during the same thread
- user frustration escalation in some conversations
- typos, informal wording, and mildly messy user language
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

DEFAULT_SEED = 42

USER_NAMES = [
    "Alex Morgan", "Priya Shah", "Daniel Kim", "Maria Chen", "Jordan Lee",
    "Taylor Brown", "Avery Johnson", "Sofia Martinez", "Noah Wilson", "Emma Clark"
]

AGENT_NAMES = [
    "IT Service Desk", "Chris Rivera", "Morgan Patel", "Help Desk Analyst",
    "Sam Walker", "Jamie Collins", "Service Desk Team"
]

GENERIC_OPENERS = [
    "Hi, I need help with this.",
    "Hello, I am having trouble and need assistance.",
    "I am running into an issue and was hoping you could help.",
]

GENERIC_THANKS = [
    "It works now, thank you.",
    "That fixed it, thanks.",
    "Looks good now, thank you for the help.",
    "I can access it now.",
]

CLOSURE_MESSAGES = [
    "Glad to hear it. I am marking this ticket resolved.",
    "Thanks for confirming. I will close this out.",
    "Happy to help. I am resolving the ticket now.",
]

UNRESOLVED_ENDINGS = [
    "We will need to escalate this to another team for further review.",
    "I am waiting on additional information from another system before proceeding.",
    "This needs to be reviewed by the systems team, so I am escalating it.",
]

MESSY_PREFIXES = [
    "hi,",
    "hello,",
    "pls,",
    "please,",
    "",
]

MESSY_SUFFIXES = [
    "",
    " thanks",
    " thank you",
    " asap",
    " this is urgent",
]

TYPO_REPLACEMENTS = {
    "cannot": "cant",
    "can't": "cant",
    "password": "passwrod",
    "account": "acount",
    "email": "emial",
    "printer": "prnter",
    "connection": "conection",
    "access": "acess",
    "because": "becuase",
    "please": "pls",
    "thanks": "thx",
    "working": "workng",
    "issue": "isue",
    "really": "realy",
    "trying": "tryng",
    "problem": "probelm",
    "shared": "shard",
    "folder": "foler",
    "locked": "lockd",
}

USER_FRUSTRATION_MESSAGES = [
    "This is really affecting my work now.",
    "I already tried basic troubleshooting and I am still stuck.",
    "This is the third time this happened this month.",
    "I need this fixed as soon as possible.",
    "I am getting pretty frustrated because I cannot do my job.",
    "This is becoming urgent and I need an update soon.",
]

PARTIAL_RESOLUTION_MESSAGES = [
    "It is a little better now, but the issue is not fully resolved.",
    "I can access part of it now, but something is still failing.",
    "That helped somewhat, but I am still seeing problems.",
    "The main issue improved, but I still cannot use everything normally.",
]

AGENT_DELAY_MESSAGES = [
    "I am still checking another system for related errors.",
    "I have reviewed one part of this, but I want to verify another dependency first.",
    "I made one update, but I need to confirm whether there is a second issue involved.",
]

MIXED_ISSUE_INTROS = [
    "Also, I am noticing another problem that may be related.",
    "There may actually be a second issue happening at the same time.",
    "I also started seeing another error after the first issue began.",
]

EXTRA_USER_DETAILS = [
    "This started this morning.",
    "It was working yesterday.",
    "This affects both my laptop and remote access workflow.",
    "I have a deadline today.",
    "Several retries did not help.",
]

EXTRA_AGENT_QUESTIONS = [
    "Can you also confirm whether this affects any other system or only this one?",
    "Did anything change recently on your machine or account before this started?",
    "Can you tell me whether this is happening only on one device?",
]

EXTRA_AGENT_ACTIONS = [
    "I checked another related setting on the account as well.",
    "I reviewed the recent changes associated with your profile.",
    "I tested the configuration from our side before proceeding.",
]

ISSUES: List[Dict] = [
    {
        "category": "account_locked",
        "titles": [
            "Cannot log in",
            "Account locked",
            "Unable to access account",
            "Sign-in issue",
        ],
        "user_open": [
            "I can't log into my account even though I believe my password is correct.",
            "My account says it is locked and I cannot access email or the portal.",
            "I was trying to sign in and now it says my account is locked.",
        ],
        "clarify_agent": [
            "Are you seeing a specific lockout or password error message?",
            "Can you confirm whether the message says the password is incorrect or that the account is locked?",
        ],
        "clarify_user": [
            "It specifically says the account is locked.",
            "The message says my account is locked after too many attempts.",
        ],
        "first_try_agent": [
            "Please try waiting a few minutes and then sign in again once.",
            "Please try one more sign-in attempt with the same password and let me know what happens.",
        ],
        "first_try_user_fail": [
            "I tried that and it still says the account is locked.",
            "I waited and tried again, but I still cannot sign in.",
        ],
        "diagnosis_agent": [
            "I checked the account status and found a lockout on the account.",
            "I reviewed the authentication logs and confirmed the account is in a locked state.",
        ],
        "fix_agent": [
            "I unlocked the account and you should be able to sign in now.",
            "I cleared the lockout and confirmed the account can authenticate again.",
        ],
        "resolution_summary": [
            "The account was locked after failed sign-in attempts. The lockout was cleared and access was restored.",
            "Authentication logs showed a lockout state. The account was unlocked and sign-in access was restored.",
        ],
    },
    {
        "category": "password_reset",
        "titles": [
            "Password reset help",
            "Forgot password",
            "Can't reset password",
            "Reset link not working",
        ],
        "user_open": [
            "I forgot my password and the reset link is not working for me.",
            "I need help resetting my password for my account.",
            "I cannot get the password reset email to work.",
        ],
        "clarify_agent": [
            "Are you receiving the reset email, or is the link itself failing?",
            "Can you confirm whether the password reset email arrives and what happens when you click it?",
        ],
        "clarify_user": [
            "I do get the email, but the link says it is invalid.",
            "The email arrives, but when I click the link it says the token is expired.",
        ],
        "first_try_agent": [
            "Please request a new reset email once more and try the latest link only.",
            "Please close older reset emails and use only the newest password reset link.",
        ],
        "first_try_user_fail": [
            "I tried the newest one and it still fails.",
            "I requested another reset email and the latest link still does not work.",
        ],
        "diagnosis_agent": [
            "I checked the reset status and found the previous reset token had expired.",
            "I reviewed the account and saw the reset token was no longer valid.",
        ],
        "fix_agent": [
            "I generated a fresh password reset link and it should work now.",
            "I issued a new reset email and removed the expired token from the account.",
        ],
        "resolution_summary": [
            "The prior reset token had expired. A fresh password reset link was issued and the reset flow worked again.",
            "The reset failure was caused by an invalid or expired token. A new reset link was generated and the reset process was restored.",
        ],
    },
    {
        "category": "vpn_issue",
        "titles": [
            "VPN not connecting",
            "Cannot connect to VPN from home",
            "Remote access issue",
            "VPN connection failed",
        ],
        "user_open": [
            "My VPN won't connect from home and I need access to internal systems.",
            "I am off campus and the VPN connection keeps failing.",
            "The VPN client says connection failed every time I try to connect.",
        ],
        "clarify_agent": [
            "Are you seeing the failure before sign-in, during authentication, or after it begins connecting?",
            "Can you share whether the VPN client shows an authentication error or a connection error?",
        ],
        "clarify_user": [
            "It fails after authentication and says connection failed.",
            "I can enter my credentials, but then the VPN says the connection cannot be established.",
        ],
        "first_try_agent": [
            "Please sign out of the VPN client, restart it, and try connecting once more.",
            "Please disconnect fully, reopen the VPN client, and try the connection again.",
        ],
        "first_try_user_fail": [
            "I tried that and restarted my laptop, but it still fails.",
            "I signed out and back in and the VPN still will not connect.",
        ],
        "diagnosis_agent": [
            "I checked the VPN configuration and found your client is still using an outdated profile.",
            "I reviewed the remote access configuration and found the VPN profile on your machine is expired.",
        ],
        "fix_agent": [
            "I sent the updated VPN profile and had you reconnect with it. The VPN connected successfully after that.",
            "Once the VPN client profile was updated, the connection completed successfully.",
        ],
        "resolution_summary": [
            "The VPN client was using an outdated or expired profile. After updating the VPN configuration, remote access worked.",
            "The connection failure was caused by an outdated VPN profile. Updating the client configuration restored VPN access.",
        ],
    },
    {
        "category": "email_issue",
        "titles": [
            "Cannot access email",
            "Mailbox not loading",
            "Outlook keeps asking for password",
            "Email issue",
        ],
        "user_open": [
            "My email inbox is not loading and I cannot read messages.",
            "Outlook keeps asking for credentials and my mailbox will not open.",
            "I cannot access my email from the desktop app.",
        ],
        "clarify_agent": [
            "Is this happening in the web mailbox, the desktop app, or both?",
            "Can you confirm whether this affects Outlook desktop only or webmail too?",
        ],
        "clarify_user": [
            "Webmail seems okay, but the desktop Outlook app will not open the mailbox correctly.",
            "It is mainly happening in Outlook on my computer. The web version looks fine.",
        ],
        "first_try_agent": [
            "Please fully close Outlook and reopen it once more.",
            "Please sign out of Outlook, close the app, and open it again.",
        ],
        "first_try_user_fail": [
            "I tried that and it is still asking for credentials.",
            "That did not help. Outlook still keeps prompting me and the mailbox will not load.",
        ],
        "diagnosis_agent": [
            "I checked the mailbox status and it looks healthy, so the issue appears to be with the local Outlook profile.",
            "I reviewed the account and the mailbox is healthy. The problem looks like stale local credentials in Outlook.",
        ],
        "fix_agent": [
            "After clearing the cached credentials and signing in again, Outlook opened the mailbox normally.",
            "The Outlook credential cache was cleared and the account was re-added. The mailbox now loads correctly.",
        ],
        "resolution_summary": [
            "The mailbox itself was healthy, but Outlook had stale local credentials. Clearing the cached credentials and reauthenticating restored access.",
            "The issue came from an outdated Outlook credential cache. After clearing cached credentials and re-adding the account, email access returned.",
        ],
    },
    {
        "category": "printer_issue",
        "titles": [
            "Printer offline",
            "Cannot print",
            "Printer not working",
            "Print jobs stuck",
        ],
        "user_open": [
            "I cannot print to the office printer and it shows offline.",
            "The department printer is unavailable from my computer.",
            "Printing jobs are stuck and the printer looks offline.",
        ],
        "clarify_agent": [
            "Which printer are you trying to use, and is anyone else nearby able to print to it?",
            "Can you confirm the printer name and whether this happens with all print jobs?",
        ],
        "clarify_user": [
            "It is the second-floor office printer and my jobs just sit in the queue.",
            "It is the shared department printer and the queue is stuck on my machine.",
        ],
        "first_try_agent": [
            "Please remove the stuck print jobs from the queue and try sending a small test page.",
            "Please clear the local print queue and attempt one new test print.",
        ],
        "first_try_user_fail": [
            "The test page still did not print and it still looks offline.",
            "I cleared the queue, but the printer still does not work from my computer.",
        ],
        "diagnosis_agent": [
            "I checked the printer mapping and queue state on your machine.",
            "I reviewed the printer queue and connection and found the local printer mapping had become stale.",
        ],
        "fix_agent": [
            "I removed and re-added the printer, and the test page printed successfully after that.",
            "After clearing the queue and reinstalling the printer mapping, printing succeeded.",
        ],
        "resolution_summary": [
            "The printer issue was caused by a stuck queue or stale local printer mapping. Re-adding the printer and clearing the queue restored printing.",
            "A stale local printer mapping prevented printing. After reinstalling the printer and clearing the queue, printing worked again.",
        ],
    },
    {
        "category": "permission_issue",
        "titles": [
            "No access to shared folder",
            "Permission denied",
            "Cannot open department drive",
            "Shared drive access issue",
        ],
        "user_open": [
            "I get permission denied when I open the department shared drive.",
            "I cannot access the shared folder that my team uses.",
            "The system says I do not have permission to open the project folder.",
        ],
        "clarify_agent": [
            "Can you confirm which shared folder path you are trying to access?",
            "Which folder are you opening, and did you have access to it previously?",
        ],
        "clarify_user": [
            "It is the project folder under the department shared drive and I used to have access.",
            "It is the shared team folder and I could open it before this week.",
        ],
        "first_try_agent": [
            "Please sign out and back into your computer once, then try opening the folder again.",
            "Please disconnect and reconnect to the shared drive once and let me know if access changes.",
        ],
        "first_try_user_fail": [
            "I tried that and I still get the same permission denied message.",
            "That did not help. I still cannot open the folder.",
        ],
        "diagnosis_agent": [
            "I reviewed your group membership and checked the folder permissions.",
            "I checked the access groups for that folder and found your account was missing the required security group.",
        ],
        "fix_agent": [
            "I added the correct access group to your account and the folder should open now.",
            "Once the required security group was applied to your account, access was restored.",
        ],
        "resolution_summary": [
            "The user was missing the required security group for the shared folder. Adding the proper access group restored folder access.",
            "Missing folder permissions caused the access error. The correct group membership was applied and access was restored.",
        ],
    },
]


def iso_time(ticket_index: int, step: int) -> str:
    day = 1 + (ticket_index % 27)
    hour = 8 + ((ticket_index + step) % 8)
    minute = (7 * step + 11 * ticket_index) % 60
    return f"2026-02-{day:02d}T{hour:02d}:{minute:02d}:00Z"


def pick(seq: List[str], rng: random.Random) -> str:
    return rng.choice(seq)


def maybe_typo_word(word: str, rng: random.Random, typo_rate: float = 0.18) -> str:
    stripped = word.strip(".,!?;:")
    lowered = stripped.lower()

    if lowered in TYPO_REPLACEMENTS and rng.random() < typo_rate:
        replacement = TYPO_REPLACEMENTS[lowered]

        if stripped.istitle():
            replacement = replacement.capitalize()
        elif stripped.isupper():
            replacement = replacement.upper()

        punctuation_suffix = word[len(stripped):] if word.startswith(stripped) else ""
        if word and word[-1] in ".,!?;:":
            punctuation_suffix = word[-1]
        return replacement + punctuation_suffix

    return word


def make_messy_text(text: str, rng: random.Random, enabled: bool = True) -> str:
    if not enabled:
        return text

    words = text.split()
    words = [maybe_typo_word(w, rng) for w in words]

    if rng.random() < 0.35 and words:
        idx = rng.randrange(len(words))
        words[idx] = words[idx].rstrip(".,!?")

    result = " ".join(words)

    if rng.random() < 0.25:
        prefix = rng.choice(MESSY_PREFIXES)
        if prefix:
            result = f"{prefix} {result}"

    if rng.random() < 0.25:
        suffix = rng.choice(MESSY_SUFFIXES)
        result = f"{result}{suffix}"

    if rng.random() < 0.12:
        result = result.lower()

    return result.strip()


def add_message(messages: List[Dict], author_name: str, author_role: str, created_at: str, text: str) -> None:
    messages.append({
        "author_name": author_name,
        "author_role": author_role,
        "created_at": created_at,
        "text": text,
    })


def pick_secondary_issue(primary_issue: Dict, rng: random.Random) -> Dict:
    candidates = [issue for issue in ISSUES if issue["category"] != primary_issue["category"]]
    return rng.choice(candidates)


def build_resolved_thread(issue: Dict, ticket_index: int, rng: random.Random) -> Tuple[List[Dict], str, Optional[str]]:
    user_name = rng.choice(USER_NAMES)
    agent_name = rng.choice(AGENT_NAMES)

    messages: List[Dict] = []
    step = 0

    has_mixed_issue = rng.random() < 0.18
    has_partial_resolution_phase = rng.random() < 0.22
    has_long_conversation = rng.random() < 0.45
    has_frustration = rng.random() < 0.35

    secondary_issue_category: Optional[str] = None

    opening = pick(issue["user_open"], rng)
    if rng.random() < 0.40:
        opening = f"{opening} {rng.choice(EXTRA_USER_DETAILS)}"
    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(opening, rng))
    step += 1

    if rng.random() < 0.55:
        add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(GENERIC_OPENERS, rng))
        step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["clarify_agent"], rng))
    step += 1

    clarify_reply = pick(issue["clarify_user"], rng)
    if rng.random() < 0.35:
        clarify_reply = f"{clarify_reply} {rng.choice(EXTRA_USER_DETAILS)}"
    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(clarify_reply, rng))
    step += 1

    if has_long_conversation and rng.random() < 0.65:
        add_message(messages, agent_name, "agent", iso_time(ticket_index, step), rng.choice(EXTRA_AGENT_QUESTIONS))
        step += 1
        add_message(
            messages,
            user_name,
            "requester",
            iso_time(ticket_index, step),
            make_messy_text(rng.choice(EXTRA_USER_DETAILS), rng),
        )
        step += 1

    if has_frustration:
        add_message(
            messages,
            user_name,
            "requester",
            iso_time(ticket_index, step),
            make_messy_text(rng.choice(USER_FRUSTRATION_MESSAGES), rng),
        )
        step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["first_try_agent"], rng))
    step += 1

    add_message(
        messages,
        user_name,
        "requester",
        iso_time(ticket_index, step),
        make_messy_text(pick(issue["first_try_user_fail"], rng), rng),
    )
    step += 1

    if has_long_conversation and rng.random() < 0.60:
        add_message(messages, agent_name, "agent", iso_time(ticket_index, step), rng.choice(EXTRA_AGENT_ACTIONS))
        step += 1
        add_message(messages, agent_name, "agent", iso_time(ticket_index, step), rng.choice(AGENT_DELAY_MESSAGES))
        step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["diagnosis_agent"], rng))
    step += 1

    mixed_summary = ""
    if has_mixed_issue:
        secondary_issue = pick_secondary_issue(issue, rng)
        secondary_issue_category = secondary_issue["category"]

        add_message(
            messages,
            user_name,
            "requester",
            iso_time(ticket_index, step),
            make_messy_text(
                f"{rng.choice(MIXED_ISSUE_INTROS)} {pick(secondary_issue['user_open'], rng)}",
                rng,
            ),
        )
        step += 1

        add_message(
            messages,
            agent_name,
            "agent",
            iso_time(ticket_index, step),
            pick(secondary_issue["clarify_agent"], rng),
        )
        step += 1

        add_message(
            messages,
            user_name,
            "requester",
            iso_time(ticket_index, step),
            make_messy_text(pick(secondary_issue["clarify_user"], rng), rng),
        )
        step += 1

        mixed_summary = f" A secondary issue related to {secondary_issue_category} was also reported during the thread."

    if rng.random() < 0.60:
        add_message(
            messages,
            agent_name,
            "agent",
            iso_time(ticket_index, step),
            rng.choice([
                "I am applying the change now.",
                "I made the needed update on my side.",
                "I completed the first corrective step.",
            ]),
        )
        step += 1

    if has_partial_resolution_phase:
        add_message(
            messages,
            agent_name,
            "agent",
            iso_time(ticket_index, step),
            "I made an initial change. Please test again and let me know whether the behavior improved.",
        )
        step += 1

        add_message(
            messages,
            user_name,
            "requester",
            iso_time(ticket_index, step),
            make_messy_text(rng.choice(PARTIAL_RESOLUTION_MESSAGES), rng),
        )
        step += 1

        add_message(
            messages,
            agent_name,
            "agent",
            iso_time(ticket_index, step),
            "Understood. I found there was an additional related setting involved, so I am making one more change.",
        )
        step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["fix_agent"], rng))
    step += 1

    add_message(
        messages,
        user_name,
        "requester",
        iso_time(ticket_index, step),
        make_messy_text(pick(GENERIC_THANKS, rng), rng, enabled=False),
    )
    step += 1

    if rng.random() < 0.75:
        add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(CLOSURE_MESSAGES, rng))
        step += 1

    resolution_summary = pick(issue["resolution_summary"], rng)
    if has_partial_resolution_phase:
        resolution_summary += " The issue required more than one corrective step before full resolution."
    if has_frustration:
        resolution_summary += " The user expressed increasing urgency during the ticket."
    if mixed_summary:
        resolution_summary += mixed_summary

    return messages, resolution_summary, secondary_issue_category


def build_unresolved_thread(issue: Dict, ticket_index: int, rng: random.Random) -> Tuple[List[Dict], Optional[str]]:
    user_name = rng.choice(USER_NAMES)
    agent_name = rng.choice(AGENT_NAMES)

    messages: List[Dict] = []
    step = 0

    secondary_issue_category: Optional[str] = None
    has_mixed_issue = rng.random() < 0.12

    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(pick(issue["user_open"], rng), rng))
    step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["clarify_agent"], rng))
    step += 1

    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(pick(issue["clarify_user"], rng), rng))
    step += 1

    if rng.random() < 0.5:
        add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(rng.choice(USER_FRUSTRATION_MESSAGES), rng))
        step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["first_try_agent"], rng))
    step += 1

    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(pick(issue["first_try_user_fail"], rng), rng))
    step += 1

    if has_mixed_issue:
        secondary_issue = pick_secondary_issue(issue, rng)
        secondary_issue_category = secondary_issue["category"]
        add_message(
            messages,
            user_name,
            "requester",
            iso_time(ticket_index, step),
            make_messy_text(
                f"{rng.choice(MIXED_ISSUE_INTROS)} {pick(secondary_issue['user_open'], rng)}",
                rng,
            ),
        )
        step += 1

    if rng.random() < 0.35:
        add_message(messages, agent_name, "agent", iso_time(ticket_index, step), rng.choice(AGENT_DELAY_MESSAGES))
        step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(UNRESOLVED_ENDINGS, rng))
    return messages, secondary_issue_category


def build_partially_resolved_thread(issue: Dict, ticket_index: int, rng: random.Random) -> Tuple[List[Dict], str, Optional[str]]:
    user_name = rng.choice(USER_NAMES)
    agent_name = rng.choice(AGENT_NAMES)

    messages: List[Dict] = []
    step = 0

    secondary_issue_category: Optional[str] = None
    has_mixed_issue = rng.random() < 0.15

    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(pick(issue["user_open"], rng), rng))
    step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["clarify_agent"], rng))
    step += 1

    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(pick(issue["clarify_user"], rng), rng))
    step += 1

    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(rng.choice(USER_FRUSTRATION_MESSAGES), rng))
    step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), pick(issue["first_try_agent"], rng))
    step += 1

    add_message(messages, user_name, "requester", iso_time(ticket_index, step), make_messy_text(rng.choice(PARTIAL_RESOLUTION_MESSAGES), rng))
    step += 1

    if has_mixed_issue:
        secondary_issue = pick_secondary_issue(issue, rng)
        secondary_issue_category = secondary_issue["category"]
        add_message(
            messages,
            user_name,
            "requester",
            iso_time(ticket_index, step),
            make_messy_text(
                f"{rng.choice(MIXED_ISSUE_INTROS)} {pick(secondary_issue['user_open'], rng)}",
                rng,
            ),
        )
        step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), rng.choice(AGENT_DELAY_MESSAGES))
    step += 1

    add_message(messages, agent_name, "agent", iso_time(ticket_index, step), rng.choice(UNRESOLVED_ENDINGS))
    step += 1

    partial_summary = (
        "The issue showed partial improvement after initial troubleshooting, "
        "but it was not fully resolved and required escalation."
    )
    if secondary_issue_category:
        partial_summary += f" A secondary issue related to {secondary_issue_category} was also reported."

    return messages, partial_summary, secondary_issue_category


def build_description(messages: List[Dict], title: str) -> str:
    opener = messages[0]["text"]
    extra = " ".join(m["text"] for m in messages[1:3]) if len(messages) > 2 else ""
    return f"{title}. Reported problem: {opener} Context: {extra}"


def generate_ticket(ticket_index: int, unresolved_rate: float, rng: random.Random) -> Dict:
    issue = rng.choice(ISSUES)
    title = pick(issue["titles"], rng)

    unresolved = rng.random() < unresolved_rate
    partially_resolved = unresolved and (rng.random() < 0.35)

    if partially_resolved:
        messages, partial_summary, secondary_issue_category = build_partially_resolved_thread(issue, ticket_index, rng)
        status = rng.choice(["In Progress", "Pending"])
        ground_truth = {
            "primary_issue_category": issue["category"],
            "secondary_issue_category": secondary_issue_category,
            "resolution_summary": partial_summary,
            "should_extract": False,
            "resolution_state": "partial",
        }
    elif unresolved:
        messages, secondary_issue_category = build_unresolved_thread(issue, ticket_index, rng)
        status = rng.choice(["Open", "In Progress", "Pending"])
        ground_truth = {
            "primary_issue_category": issue["category"],
            "secondary_issue_category": secondary_issue_category,
            "resolution_summary": None,
            "should_extract": False,
            "resolution_state": "unresolved",
        }
    else:
        messages, resolution_summary, secondary_issue_category = build_resolved_thread(issue, ticket_index, rng)
        status = "Closed"
        ground_truth = {
            "primary_issue_category": issue["category"],
            "secondary_issue_category": secondary_issue_category,
            "resolution_summary": resolution_summary,
            "should_extract": True,
            "resolution_state": "resolved",
        }

    description = build_description(messages, title)

    return {
        "ticket_id": f"TDX-{10000 + ticket_index}",
        "status": status,
        "title": title,
        "description": description,
        "messages": messages,
        "ground_truth": ground_truth,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate realistic synthetic TeamsDynamix-style tickets.")
    parser.add_argument("--count", type=int, default=200, help="Number of tickets to generate")
    parser.add_argument("--output", default="simulated_tdx_tickets.jsonl", help="Output JSONL path")
    parser.add_argument("--unresolved-rate", type=float, default=0.15, help="Fraction of tickets that remain unresolved")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for reproducible output")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        generate_ticket(i, unresolved_rate=args.unresolved_rate, rng=rng)
        for i in range(1, args.count + 1)
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    resolved = sum(1 for row in rows if row["ground_truth"]["resolution_state"] == "resolved")
    partial = sum(1 for row in rows if row["ground_truth"]["resolution_state"] == "partial")
    unresolved_count = sum(1 for row in rows if row["ground_truth"]["resolution_state"] == "unresolved")
    mixed = sum(1 for row in rows if row["ground_truth"]["secondary_issue_category"] is not None)

    print(f"Wrote {len(rows)} tickets to {out_path}")
    print(f"Resolved tickets:          {resolved}")
    print(f"Partially resolved tickets:{partial}")
    print(f"Unresolved tickets:        {unresolved_count}")
    print(f"Mixed-issue tickets:       {mixed}")


if __name__ == "__main__":
    main()