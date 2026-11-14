"""Offline research review only. No status grants paper or live authority."""
from dataclasses import dataclass
from enum import Enum
from typing import Literal

from ._validation import member, text


class RiskStatus(str, Enum):
    REJECT = "REJECT"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    PASS_RESEARCH_ONLY = "PASS_RESEARCH_ONLY"


@dataclass(frozen=True, kw_only=True)
class RiskDecision:
    scope: str
    candidate: str
    rule_versions: tuple[str, ...]
    status: RiskStatus
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        text(self.scope, "scope")
        text(self.candidate, "candidate")
        member(self.status, RiskStatus)
        for name in ("rule_versions", "reasons"):
            values = tuple(getattr(self, name))
            if not values:
                raise ValueError(f"{name} cannot be empty")
            for value in values:
                text(value, name)
            object.__setattr__(self, name, values)

    @property
    def live_order_authorized(self) -> Literal[False]:
        return False

    def __bool__(self) -> bool:
        raise TypeError("Inspect research status explicitly; RiskDecision is not order approval")

    def require_execution_authorization(self, mode: str) -> None:
        """Fail closed for every mode, including dry-run/paper; no dispatch exists."""
        raise PermissionError(f"{mode}: NOT_AUTHORIZED; research decisions never authorize orders")
