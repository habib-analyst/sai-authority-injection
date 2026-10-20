"""Typed operator overrides with schema validation.

SAI routes operator corrections through typed, schema-validated tool-call-shaped
payloads instead of free text. Three override classes carry distinct precedence:

    CONSTRAINT (rank 100) > PRIORITY (rank 50) > STRATEGY (rank 25)

Validation is the first line of defense: ambiguous or malformed operator input is
rejected *before* it can enter the authority channel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class OverrideClass(str, Enum):
    CONSTRAINT = "constraint"
    PRIORITY = "priority"
    STRATEGY = "strategy"


# Precedence table: higher rank always outranks lower rank, regardless of
# recency. This is what makes authority "directed" rather than positional.
PRECEDENCE = {
    OverrideClass.CONSTRAINT: 100,
    OverrideClass.PRIORITY: 50,
    OverrideClass.STRATEGY: 25,
}

_TARGET_RE = re.compile(r"^[a-z][a-z0-9_: ]{1,78}[a-z0-9_]$")
_REQUIRED_KEYS = ("class", "target", "directive")
_MAX_DIRECTIVE_LEN = 280
_MIN_DIRECTIVE_LEN = 10


def validate_override(payload) -> tuple[bool, str]:
    """Schema-validate a raw operator override payload.

    Returns (True, "ok") or (False, reason). Never raises on bad input.
    """
    if not isinstance(payload, dict):
        return False, "payload must be a JSON object"
    for key in _REQUIRED_KEYS:
        if key not in payload:
            return False, f"missing required key: {key}"
    cls_raw = payload["class"]
    try:
        cls = OverrideClass(str(cls_raw).lower())
    except ValueError:
        return False, f"unknown override class: {cls_raw!r} (want constraint|priority|strategy)"
    target = payload["target"]
    if not isinstance(target, str) or not _TARGET_RE.match(target):
        return False, f"invalid target: {target!r} (lowercase, 3-80 chars, a-z 0-9 _ : space)"
    directive = payload["directive"]
    if not isinstance(directive, str):
        return False, "directive must be a string"
    directive = directive.strip()
    if len(directive) < _MIN_DIRECTIVE_LEN:
        return False, f"directive too short ({len(directive)} chars, min {_MIN_DIRECTIVE_LEN})"
    if len(directive) > _MAX_DIRECTIVE_LEN:
        return False, f"directive too long ({len(directive)} chars, max {_MAX_DIRECTIVE_LEN})"
    return True, "ok"


@dataclass(frozen=True)
class Override:
    """A registered, validated operator override.

    Frozen: once registered, neither the operator's chat text nor the planner
    can mutate it. Only a newer override of higher-or-equal rank can supersede.
    """

    cls: OverrideClass
    target: str
    directive: str
    rank: int
    override_id: str
    seq: int  # registration order; breaks ties within the same rank

    def render(self) -> str:
        return (
            f"[AUTHORITY rank={self.rank} id={self.override_id}] "
            f"{self.cls.value.upper()} on {self.target}: {self.directive}"
        )
