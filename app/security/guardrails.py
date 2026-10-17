import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"(reveal|show|print|dump)\s+(the\s+)?(system prompt|hidden prompt|developer message)",
    r"you are now (dan|an unrestricted|jailbroken)",
    r"bypass\s+(all\s+)?(safety|security|guardrails)",
    r"disregard\s+(all\s+)?(rules|instructions)",
    r"<\s*/?\s*system\s*>",
]
UNSAFE_PATTERNS = [
    r"\b(make|build|synthesize)\b.{0,50}\b(bomb|explosive|poison)\b",
    r"\b(write|deploy)\b.{0,50}\b(ransomware|credential stealer|keylogger)\b",
    r"\bsteal\b.{0,30}\b(passwords|credentials|session tokens)\b",
]
PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\w)(?=(?:\D*\d){10,})\+?\d[\d .()\-]{7,}\d(?!\w)"), "[REDACTED_PHONE]"),
    (re.compile(r"\b(?:sk-[A-Za-z0-9_-]{20,}|gsk_[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})\b"), "[REDACTED_SECRET]"),
]


@dataclass(frozen=True)
class GuardrailResult:
    allowed: bool
    text: str
    reason: str | None = None
    redactions: int = 0


class GuardrailEngine:
    """Deterministic input/output filters; not a substitute for a full policy classifier."""

    def inspect_input(self, text: str) -> GuardrailResult:
        for pattern in INJECTION_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                return GuardrailResult(False, "", reason="prompt_injection")
        for pattern in UNSAFE_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL):
                return GuardrailResult(False, "", reason="unsafe_request")
        return self.redact(text)

    def redact(self, text: str) -> GuardrailResult:
        count = 0
        for pattern, replacement in PII_PATTERNS:
            text, substitutions = pattern.subn(replacement, text)
            count += substitutions
        return GuardrailResult(True, text, redactions=count)

    def inspect_output(self, text: str) -> str:
        return self.redact(text).text
