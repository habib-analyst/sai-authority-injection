# SAI: Structured Authority Injection — implementation writeup

Project #51 · Habib Ur Rehman · 2026-10-20
Paper implemented: arXiv 2610.22373 (Tran et al.)

## Claim

Operator corrections delivered as **typed, schema-validated overrides**
through a **directed authority channel** persist across long agent episodes,
while free-text corrections erode. Three override classes
(Constraint / Priority / Strategy) carry fixed precedence ranks, and a ranked
authority context prepended to each prompt outranks recency. No fine-tuning.

## Method

Deterministic policy-simulator study (CPU-only, zero LLM calls):

- **Planner policy (pure function):** each instruction scores
  `authority_rank × 1000 + 0.85^(turn − injection_turn)` + seeded tie-break
  jitter. The planner sees its own context plus the orchestrator-prepended
  authority snapshot; it holds no reference to the store and no mutation API
  (structural channel separation, covered by tests).
- **Free-text arm:** the operator directive is a plain turn-0 chat message
  (rank 0). Each turn the planner internalizes the winner into a plan note,
  so once a distractor wins, the violation propagates in the agent's own plan.
- **SAI arm:** the same directive is registered once as a typed override
  (Constraint rank 100 / Priority 50 / Strategy 25) via the operator-only
  handle; the ranked snapshot is prepended every turn.
- **Third arm:** free-text directive re-sent every 3 turns (the realistic
  human workaround).
- **8 failure-mode scenarios** (scope drift, forbidden tool, budget cap, data
  boundary, priority reorder, output contract, strategy shift, retry policy),
  distractors at turns 2/4/6/8, 300 seeded episodes per scenario per arm,
  10 turns per episode. Distractor schedule, decay, and gate bars were fixed
  a priori.

## Measured numbers

All from `experiments/run_gates.py` (fixed seeds, reproducible):

- **Gate 2 — erosion (disease measured first):**
  compliance(turn 0) − compliance(turn 9), free-text = **1.000**,
  paired bootstrap 95% CI **[1.000, 1.000]** (bar: ≥ 0.20, CI excludes 0).
  Free-text curve: `[1.0, 1.0, 0.0, 0.0, ...]` — honored 2 turns, then
  captured permanently. Mean episode compliance 0.200.
- **Gate 3 — load-bearing margin:** mean(SAI) − mean(free-text) = **0.800**,
  95% CI **[0.800, 0.800]** (bar: ≥ 0.15, CI excludes 0).
  Mean episode compliance: SAI **1.000** vs free-text **0.200**;
  SAI curve is `[1.0] × 10` in all 8 scenarios.
- **Overhead:** prompt tokens/episode — SAI **1047.3** vs free-text-once
  1088.7 vs re-sent reminders **1233.5**. SAI is cheapest *and* fully
  compliant; reminders recover only 0.502 compliance at ~18% extra tokens
  and race distractors at equal recency (turn-6 compliance 0.51).
- **Schema validation:** 8/8 well-formed overrides accepted;
  50/50 malformed payloads rejected (bad class, bad target, short/long
  directive, missing keys, non-dict input).
- **Tests:** 20/20 green (`pytest tests/`).

## Limits (stated plainly)

1. **Simulator, not LLMs.** The paper evaluated 9 LLMs; this build uses a
   deterministic policy simulator. The numbers demonstrate the mechanism
   (ranked directed authority persists; positional free-text authority
   erodes), not LLM generalization of the exact margins.
2. **Modeling choices are fixed a priori but still choices:** decay 0.85,
   distractor turns {2,4,6,8}, plan-note internalization. Different choices
   shift absolute numbers; the rank-vs-recency structural claim does not
   depend on them.
3. **Single-override episodes.** Multi-override precedence (Constraint >
   Priority > Strategy) is unit-tested for ranking but not measured in long
   episodes with conflicting operator directives.
4. **No adversarial planner.** The planner is cooperative-but-forgetful; a
   planner actively trying to evade authority is out of scope.
5. **MCP wire format.** Overrides are tool-call-*shaped* dicts validated
   against the class schemas; no live MCP server transport is implemented.

## Reproduce

```bash
python3 -m pytest tests/ -q
python3 experiments/run_gates.py   # prints the full JSON summary
python3 demo/demo.py               # 30-second visual
```
