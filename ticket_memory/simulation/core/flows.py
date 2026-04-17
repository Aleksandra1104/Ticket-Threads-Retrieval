from __future__ import annotations

import random
from typing import List

from .models import ConversationResult, Message, Scenario
from .renderer import apply_persona
from .utils import build_description, choose, iso_time

GENERIC_OPENERS = [
    "Hi, I can help with this.",
    "Thanks for reaching out. I am looking into it now.",
    "I can help troubleshoot that with you.",
]
GENERIC_THANKS = [
    "It works now, thank you.",
    "That fixed it, thanks.",
    "Looks good now. I can use it again.",
    "I can access it now.",
]
CLOSURE_MESSAGES = [
    "Thanks for confirming. I will close this ticket.",
    "Glad to hear it. I am marking this resolved.",
    "Happy to help. I am resolving the ticket now.",
]
UNRESOLVED_ENDINGS = [
    "I need to escalate this to another team for further review.",
    "This still requires back-end review, so I am escalating it.",
    "I am escalating this because the initial troubleshooting did not fully resolve it.",
]
FRUSTRATION_MESSAGES = [
    "This is blocking my work right now.",
    "I already tried the basic steps and I am still stuck.",
    "This has happened more than once and it is getting frustrating.",
    "I have a deadline today, so I need this fixed soon.",
]
PARTIAL_MESSAGES = [
    "It is a little better now, but the issue is not fully resolved.",
    "That helped somewhat, but I am still seeing problems.",
    "The main issue improved, but I still cannot use everything normally.",
]
AGENT_DELAY_MESSAGES = [
    "I am checking another related dependency before I confirm the fix.",
    "I made one update, but I want to verify another setting as well.",
    "I am seeing signs that there may be a second contributing issue.",
]
MIXED_ISSUE_INTROS = [
    "Also, I am noticing another problem that may be related.",
    "There may actually be a second issue happening at the same time.",
    "I also started seeing another error after the first issue began.",
]
EXTRA_AGENT_QUESTIONS = [
    "Can you confirm whether this affects one device or several?",
    "Did anything change recently before this started?",
    "Is the same behavior happening in the web version or only the desktop client?",
]
EXTRA_AGENT_ACTIONS = [
    "I checked the related account settings as well.",
    "I reviewed the recent configuration changes tied to this service.",
    "I tested the configuration from our side before applying the next step.",
]


def _message(messages: List[Message], scenario: Scenario, step: int, role: str, text: str, rng: random.Random, allow_typos: bool) -> None:
    name = scenario.requester_name if role == "requester" else scenario.agent_name
    persona = scenario.user_persona if role == "requester" else scenario.agent_persona
    messages.append(
        Message(
            author_name=name,
            author_role=role,
            created_at=iso_time(int(scenario.ticket_id.split("-")[-1]), step),
            text=apply_persona(text, persona, rng, allow_typos=allow_typos),
        )
    )


def _resolved_summary(scenario: Scenario, rng: random.Random, include_partial_note: bool, include_mixed_note: bool) -> str:
    summary = choose(scenario.issue_variant.resolution_summary, rng)
    if include_partial_note:
        summary += " The issue required more than one corrective step before full resolution."
    if scenario.user_persona.name == "frustrated":
        summary += " The user expressed urgency during the conversation."
    if include_mixed_note and scenario.secondary_issue_variant is not None:
        summary += f" A secondary issue related to {scenario.secondary_issue_variant.family} was also reported."
    return summary


def _base_opening(messages: List[Message], scenario: Scenario, rng: random.Random, step: int) -> int:
    opening = choose(scenario.issue_variant.user_openers, rng)
    if scenario.issue_variant.environment_details and rng.random() < 0.55:
        opening = f"{opening} {choose(scenario.issue_variant.environment_details, rng)}"
    _message(messages, scenario, step, "requester", opening, rng, allow_typos=True)
    step += 1
    if rng.random() < 0.50:
        _message(messages, scenario, step, "agent", choose(GENERIC_OPENERS, rng), rng, allow_typos=False)
        step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.clarify_agent, rng), rng, allow_typos=False)
    step += 1
    clarify = choose(scenario.issue_variant.clarify_user, rng)
    if scenario.issue_variant.signals and rng.random() < 0.50:
        clarify = f"{clarify} {choose(scenario.issue_variant.signals, rng)}"
    _message(messages, scenario, step, "requester", clarify, rng, allow_typos=True)
    return step + 1


def direct_resolution(scenario: Scenario, rng: random.Random) -> ConversationResult:
    messages: List[Message] = []
    step = _base_opening(messages, scenario, rng, 0)
    if rng.random() < 0.45:
        _message(messages, scenario, step, "agent", choose(EXTRA_AGENT_ACTIONS, rng), rng, allow_typos=False)
        step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.diagnosis_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.fix_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "requester", choose(GENERIC_THANKS, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "agent", choose(CLOSURE_MESSAGES, rng), rng, allow_typos=False)
    description = build_description(messages[0].text, scenario.title, [m.text for m in messages[1:3]])
    return ConversationResult(scenario, description, messages, _resolved_summary(scenario, rng, False, False), "resolved", True, 0.97)


def failed_first_fix_then_success(scenario: Scenario, rng: random.Random) -> ConversationResult:
    messages: List[Message] = []
    step = _base_opening(messages, scenario, rng, 0)
    if rng.random() < 0.45:
        _message(messages, scenario, step, "requester", choose(FRUSTRATION_MESSAGES, rng), rng, allow_typos=True)
        step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.first_try_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "requester", choose(scenario.issue_variant.first_try_user_fail, rng), rng, allow_typos=True)
    step += 1
    if rng.random() < 0.55:
        _message(messages, scenario, step, "agent", choose(EXTRA_AGENT_QUESTIONS, rng), rng, allow_typos=False)
        step += 1
        if scenario.issue_variant.environment_details:
            _message(messages, scenario, step, "requester", choose(scenario.issue_variant.environment_details, rng), rng, allow_typos=True)
            step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.diagnosis_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.fix_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "requester", choose(GENERIC_THANKS, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "agent", choose(CLOSURE_MESSAGES, rng), rng, allow_typos=False)
    description = build_description(messages[0].text, scenario.title, [m.text for m in messages[1:3]])
    return ConversationResult(scenario, description, messages, _resolved_summary(scenario, rng, False, False), "resolved", True, 0.94)


def escalation(scenario: Scenario, rng: random.Random) -> ConversationResult:
    messages: List[Message] = []
    step = _base_opening(messages, scenario, rng, 0)
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.first_try_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "requester", choose(scenario.issue_variant.first_try_user_fail, rng), rng, allow_typos=True)
    step += 1
    _message(messages, scenario, step, "agent", choose(AGENT_DELAY_MESSAGES, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "agent", choose(UNRESOLVED_ENDINGS, rng), rng, allow_typos=False)
    description = build_description(messages[0].text, scenario.title, [m.text for m in messages[1:3]])
    return ConversationResult(scenario, description, messages, None, "unresolved", False, 0.55)


def partial_resolution(scenario: Scenario, rng: random.Random) -> ConversationResult:
    messages: List[Message] = []
    step = _base_opening(messages, scenario, rng, 0)
    _message(messages, scenario, step, "requester", choose(FRUSTRATION_MESSAGES, rng), rng, allow_typos=True)
    step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.first_try_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "requester", choose(PARTIAL_MESSAGES, rng), rng, allow_typos=True)
    step += 1
    _message(messages, scenario, step, "agent", choose(AGENT_DELAY_MESSAGES, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "agent", choose(UNRESOLVED_ENDINGS, rng), rng, allow_typos=False)
    description = build_description(messages[0].text, scenario.title, [m.text for m in messages[1:3]])
    summary = "The issue showed partial improvement after initial troubleshooting, but it was not fully resolved and required escalation."
    return ConversationResult(scenario, description, messages, summary, "partial", False, 0.72)


def mixed_issue(scenario: Scenario, rng: random.Random) -> ConversationResult:
    messages: List[Message] = []
    step = _base_opening(messages, scenario, rng, 0)
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.first_try_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "requester", choose(scenario.issue_variant.first_try_user_fail, rng), rng, allow_typos=True)
    step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.diagnosis_agent, rng), rng, allow_typos=False)
    step += 1
    if scenario.secondary_issue_variant is not None:
        follow_on = f"{choose(MIXED_ISSUE_INTROS, rng)} {choose(scenario.secondary_issue_variant.user_openers, rng)}"
        _message(messages, scenario, step, "requester", follow_on, rng, allow_typos=True)
        step += 1
        _message(messages, scenario, step, "agent", choose(scenario.secondary_issue_variant.clarify_agent, rng), rng, allow_typos=False)
        step += 1
        _message(messages, scenario, step, "requester", choose(scenario.secondary_issue_variant.clarify_user, rng), rng, allow_typos=True)
        step += 1
    _message(messages, scenario, step, "agent", choose(scenario.issue_variant.fix_agent, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "requester", choose(GENERIC_THANKS, rng), rng, allow_typos=False)
    step += 1
    _message(messages, scenario, step, "agent", choose(CLOSURE_MESSAGES, rng), rng, allow_typos=False)
    description = build_description(messages[0].text, scenario.title, [m.text for m in messages[1:3]])
    return ConversationResult(scenario, description, messages, _resolved_summary(scenario, rng, False, True), "resolved", True, 0.91)


FLOW_HANDLERS = {
    "direct_resolution": direct_resolution,
    "failed_first_fix_then_success": failed_first_fix_then_success,
    "escalation": escalation,
    "partial_resolution": partial_resolution,
    "mixed_issue": mixed_issue,
}


