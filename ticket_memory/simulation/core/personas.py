from __future__ import annotations

from .models import Persona


USER_PERSONAS = [
    Persona("vague", "brief and incomplete", ["Hi,", "Hello,", ""], ["", " thanks", " can you help"], ["omit exact error details"]),
    Persona("frustrated", "stressed and urgent", ["Please,", "Hi,", ""], ["", " this is urgent", " I really need this fixed"], ["mention work impact"]),
    Persona("technical", "specific and detail-heavy", ["Hello,", ""], ["", " for reference"], ["include exact symptoms"]),
    Persona("cooperative", "polite and responsive", ["Hi team,", "Hello,", ""], ["", " thank you", " appreciate the help"], ["answer questions directly"]),
]


AGENT_PERSONAS = [
    Persona("concise", "short and direct", ["", "Hi,", "Hello,"], ["", " let me know what you see"], ["keep steps short"]),
    Persona("methodical", "structured and diagnostic", ["", "Thanks for the details."], ["", " once you test that, reply with the result"], ["verify symptoms"]),
    Persona("empathetic", "supportive and reassuring", ["", "I understand the disruption."], ["", " I know this is blocking work"], ["acknowledge impact"]),
]


