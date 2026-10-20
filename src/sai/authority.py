"""The directed authority channel.

Only the operator holds a registration handle. The planner NEVER receives the
store itself, only an immutable ranked snapshot that the orchestrator prepends
to each prompt. There is deliberately no API for the planner to add, remove, or
reorder entries: the channel is directed operator -> planner, never the reverse.
"""
from __future__ import annotations

from dataclasses import dataclass

from .overrides import Override, OverrideClass, PRECEDENCE, validate_override


class _ChannelToken:
    """Unforgeable capability token. Only the store mints these."""

    __slots__ = ()


@dataclass(frozen=True)
class AuthoritySnapshot:
    """Immutable, precedence-ranked view of active overrides.

    Frozen dataclass holding a tuple: planner-side code cannot append, remove,
    or reorder. Rendered text is what gets prepended to each prompt.
    """

    entries: tuple[Override, ...]

    def render(self) -> str:
        if not self.entries:
            return ""
        lines = ["AUTHORITY CONTEXT (operator directives, binding):"]
        lines.extend("  " + e.render() for e in self.entries)
        return "\n".join(lines)


class OperatorHandle:
    """Capability proving the caller is the operator side of the channel."""

    __slots__ = ("_token",)

    def __init__(self, token: _ChannelToken):
        self._token = token


class AuthorityStore:
    """Operator-side registry. Lives on the orchestrator, never in planner scope."""

    def __init__(self):
        self._entries: list[Override] = []
        self._seq = 0
        self._token = _ChannelToken()

    def operator_handle(self) -> OperatorHandle:
        """Issue the (single) operator capability. The planner never gets one."""
        return OperatorHandle(self._token)

    def register(self, handle: OperatorHandle, payload: dict) -> Override:
        """Validate and register an override. Operator-only."""
        token = getattr(handle, "_token", None)
        if token is not self._token:
            raise PermissionError("override registration requires the operator channel handle")
        ok, reason = validate_override(payload)
        if not ok:
            raise ValueError(f"override rejected by schema validation: {reason}")
        cls = OverrideClass(str(payload["class"]).lower())
        self._seq += 1
        ov = Override(
            cls=cls,
            target=payload["target"].strip(),
            directive=payload["directive"].strip(),
            rank=PRECEDENCE[cls],
            override_id=f"ov-{self._seq:04d}",
            seq=self._seq,
        )
        self._entries.append(ov)
        return ov

    def snapshot(self) -> AuthoritySnapshot:
        """Ranked snapshot for prepending. Highest rank first, then oldest first."""
        ranked = tuple(sorted(self._entries, key=lambda e: (-e.rank, e.seq)))
        return AuthoritySnapshot(entries=ranked)

    def __len__(self) -> int:
        return len(self._entries)
