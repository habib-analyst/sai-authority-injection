"""sai-authority-injection: open implementation of SAI (arXiv 2610.22373).

Structured Authority Injection: typed, schema-validated MCP-style operator
overrides with a ranked authority context and a directed authority channel
inaccessible to the planner.
"""
from .authority import AuthoritySnapshot, AuthorityStore, OperatorHandle
from .metrics import (
    compliance_curve,
    episode_means,
    episode_tokens,
    mean_compliance,
    paired_bootstrap_ci,
)
from .overrides import PRECEDENCE, Override, OverrideClass, validate_override
from .scenarios import DISTRACTOR_TURNS, EPISODE_TURNS, SCENARIOS, Scenario
from .simulator import EpisodeResult, planner_decide, run_episode

__all__ = [
    "AuthoritySnapshot", "AuthorityStore", "OperatorHandle",
    "Override", "OverrideClass", "PRECEDENCE", "validate_override",
    "SCENARIOS", "Scenario", "DISTRACTOR_TURNS", "EPISODE_TURNS",
    "EpisodeResult", "planner_decide", "run_episode",
    "compliance_curve", "episode_means", "episode_tokens",
    "mean_compliance", "paired_bootstrap_ci",
]

__version__ = "0.1.0"
