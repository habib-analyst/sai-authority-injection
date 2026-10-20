# sai-authority-injection

An open implementation of **SAI: Structured Authority Injection**
(arXiv 2610.22373 — Tran et al., "Structured Authority Injection: Persistent
Human Control for LLM Agents via Typed MCP Overrides").

Built by **Habib Ur Rehman** — [github.com/habib-analyst](https://github.com/habib-analyst)

## The problem

LLM agents lose operator instructions as context grows. A correction typed in
plain chat at turn 0 is just another message by turn 9: positional authority
decays, later instructions win, and the operator's directive silently stops
being honored. This is **authority erosion**, and it is measured, not assumed:
in our 8-scenario study the free-text baseline drops from 100% to 0%
compliance by turn 2 and never recovers.

## The fix (SAI)

Route operator corrections through **typed, schema-validated overrides** in a
**directed authority channel** the planner cannot touch:

- **3 override classes** with fixed precedence:
  `CONSTRAINT` (rank 100) > `PRIORITY` (rank 50) > `STRATEGY` (rank 25)
- **Schema validation** rejects malformed overrides before registration
  (8/8 well-formed accepted, 50/50 malformed rejected in our checks).
- **Ranked authority context** is prepended to every prompt; rank outranks
  recency, so the directive persists across planning cycles.
- **Directed channel**: only the operator holds a registration handle. The
  planner receives an immutable ranked snapshot — structurally, there is no
  API for it to add, remove, or reorder authority entries.
- **No fine-tuning**, no model weights needed. Protocol-level, CPU-only.

## 30-second demo

```bash
python3 demo/demo.py
```

```
free-text override (baseline)             [##........]  2/10 turns compliant
SAI typed override (Constraint, rank 100) [##########]  10/10 turns compliant
```

## Install and test

Zero third-party runtime dependencies (stdlib only; numpy + pytest for
experiments/tests, both commonly preinstalled).

```bash
python3 -m pytest tests/ -q          # 20 tests
python3 experiments/run_gates.py     # full measurement, fixed seeds
```

## Measured results

8 failure-mode scenarios × 300 seeded episodes × 3 arms, 10 turns each
(deterministic policy simulator; see Honesty below).

| Arm | Mean episode compliance | Prompt tokens / episode |
|---|---|---|
| Free-text override (once) | 0.200 | 1088.7 |
| Free-text re-sent every 3 turns | 0.502 | 1233.5 |
| **SAI typed override** | **1.000** | **1047.3** |

- **Gate 2 (erosion):** compliance(turn 0) − compliance(turn 9) = **1.000**,
  95% CI [1.000, 1.000].
- **Gate 3 (load-bearing margin):** SAI − free-text = **0.800**,
  95% CI [0.800, 0.800] (paired bootstrap, 5000 replicates).
- Re-sent reminders partially recover compliance but cost ~18% more tokens
  and still race distractors at equal recency (turn-6 compliance 0.51).
  SAI wins on compliance *and* overhead.

## Layout

```
src/sai/
  overrides.py    typed override classes + schema validation
  authority.py    directed authority channel (operator handle, ranked snapshots)
  scenarios.py    8 failure-mode scenarios with distractor schedules
  simulator.py    deterministic planner policy + episode runner
  metrics.py      compliance metrics + paired bootstrap CIs
demo/demo.py              30-second erosion-vs-persistence demo
experiments/run_gates.py  gate measurement (reproduces every number above)
tests/test_sai.py         20 tests
KILLGATE.md               kill-gate verdicts with measured evidence
```

## Honesty: what these numbers are

This build measures the SAI *mechanism* with a deterministic policy
simulator (seeded RNG, recency-decayed salience, plan internalization;
distractor schedule and decay fixed before any number was seen). No LLMs were
called. The paper evaluated 9 LLMs across 8 scenarios; this CPU implementation
mirrors its 8-scenario structure with a simulator and says so plainly. The
claim demonstrated here is structural: ranked, directed authority persists
where positional free-text authority erodes. LLM generalization of the exact
margins is not claimed.

## Citation

```bibtex
@article{tran2026sai,
  title   = {Structured Authority Injection: Persistent Human Control
             for LLM Agents via Typed MCP Overrides},
  author  = {Tran, Ryan and Zhang, YueMing and Tao, Lingfeng and Yang, Nan
             and Xu, Shengjie and Yin, Yu and Xu, Shuai and Zhang, Wenlu
             and Xu, Hailu},
  journal = {arXiv preprint arXiv:2610.22373},
  year    = {2026}
}
```

## License

MIT — see LICENSE.
