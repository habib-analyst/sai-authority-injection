"""Deterministic planner policy + episode runner.

The planner is a pure function: it sees (a) its own accumulated context and
(b) the authority snapshot text the orchestrator prepends. It has NO reference
to the AuthorityStore and no API to mutate the snapshot: that is the directed
channel, enforced structurally.

Decision rule (salience):
    score(instr) = authority_rank * RANK_W + decay ** (turn - inj_turn) + jitter

Free-text arm: every instruction has authority_rank 0, so recency rules and the
operator's turn-0 message erodes as context grows. The planner also
internalizes each turn's winner into a plan note, so once a distractor wins,
the agent's own plan carries the violation forward (this is the erosion
mechanism, fixed a priori).

SAI arm: the typed override sits in the snapshot with its class rank
(100/50/25). RANK_W is large, so ranked authority always outranks recency:
the directive persists no matter how long the episode runs.

Deterministic: all randomness comes from random.Random(seed); identical seeds
give identical episodes.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from .authority import AuthoritySnapshot, AuthorityStore
from .scenarios import DISTRACTOR_TURNS, EPISODE_TURNS, Scenario

RANK_W = 1000.0
DECAY = 0.85


@dataclass
class Instruction:
    text: str
    inj_turn: int
    blocks_override: bool  # True -> this instruction violates the operator directive
    authority_rank: int = 0


def planner_decide(
    context: list[Instruction],
    authority: AuthoritySnapshot,
    turn: int,
    rng: random.Random,
) -> Instruction:
    """Pure planner policy. Returns the winning instruction for this turn.

    NOTE: `authority` is a rendered snapshot (text + ranks), never the store.
    The planner cannot see, add, or remove authority entries.
    """
    best: Instruction | None = None
    best_score = float("-inf")
    for instr in context:
        score = instr.authority_rank * RANK_W + DECAY ** (turn - instr.inj_turn)
        score += rng.random() * 1e-6  # deterministic tie-break only
        if score > best_score:
            best_score, best = score, instr
    assert best is not None
    return best


@dataclass
class EpisodeResult:
    scenario_id: str
    mode: str
    seed: int
    compliance: list[bool]   # per-turn: did the winner honor the operator directive?
    tokens_per_turn: list[int]


def _words(text: str) -> int:
    return len(text.split())


def run_episode(scenario: Scenario, mode: str, seed: int,
                turns: int = EPISODE_TURNS) -> EpisodeResult:
    """Run one seeded episode. mode in {freetext, freetext_repeat, sai}."""
    if mode not in ("freetext", "freetext_repeat", "sai"):
        raise ValueError(f"unknown mode: {mode}")
    rng = random.Random(seed)
    context: list[Instruction] = []
    store = AuthorityStore()
    handle = store.operator_handle()

    compliance: list[bool] = []
    tokens: list[int] = []

    if mode == "sai":
        # Operator registers through the directed channel, once, at turn 0.
        store.register(handle, scenario.typed_payload)
    else:
        # Free-text arm: the directive is just another chat message at turn 0.
        context.append(Instruction(text=scenario.freetext_message, inj_turn=0,
                                   blocks_override=False))

    distractors = list(scenario.distractors)
    rng.shuffle(distractors)

    for t in range(turns):
        snapshot = store.snapshot()  # orchestrator-side; planner only sees the render
        # Distractors arrive on schedule (later instructions, positional advantage).
        if t in DISTRACTOR_TURNS:
            d = distractors[DISTRACTOR_TURNS.index(t) % len(distractors)]
            context.append(Instruction(text=d, inj_turn=t, blocks_override=True))
        # Repeat-reminder arm: operator re-sends the free-text directive.
        if mode == "freetext_repeat" and t > 0 and t % 3 == 0:
            context.append(Instruction(text=scenario.freetext_message, inj_turn=t,
                                       blocks_override=False))
        # SAI arm: authority entries compete with their class rank.
        candidates = list(context)
        for ov in snapshot.entries:
            candidates.append(Instruction(text=ov.render(), inj_turn=0,
                                          blocks_override=False,
                                          authority_rank=ov.rank))
        winner = planner_decide(context=candidates, authority=snapshot, turn=t, rng=rng)
        compliance.append(not winner.blocks_override)
        # Planner internalizes the winner into its ongoing plan notes.
        context.append(Instruction(
            text=f"Plan note t{t}: proceeding per: {winner.text[:60]}",
            inj_turn=t, blocks_override=winner.blocks_override))
        # Prompt tokens charged this turn: planner context + prepended authority block.
        prompt_tokens = sum(_words(i.text) for i in context) + _words(snapshot.render())
        tokens.append(prompt_tokens)

    return EpisodeResult(scenario_id=scenario.sid, mode=mode, seed=seed,
                         compliance=compliance, tokens_per_turn=tokens)
