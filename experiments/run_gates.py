#!/usr/bin/env python3
"""Gate 2 (authority-erosion sanity) and Gate 3 (load-bearing) measurement.

Runs R seeded episodes per scenario x 3 arms (freetext, freetext_repeat, sai),
then reports:
  Gate 2: erosion gap = compliance(t=0) - compliance(t=9) for the free-text arm.
          PASS bar (fixed a priori): gap >= 0.20 with paired bootstrap 95% CI
          excluding 0.
  Gate 3: margin = mean_episode_compliance(sai) - mean_episode_compliance(freetext).
          PASS bar (fixed a priori): margin >= 0.15 with paired bootstrap 95% CI
          excluding 0.
Also reports: per-arm compliance curves, total prompt tokens (overhead), and
schema-validation rejection stats.

Deterministic: seeds are fixed; re-running reproduces every number.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from sai import (  # noqa: E402
    SCENARIOS,
    episode_means,
    episode_tokens,
    paired_bootstrap_ci,
    run_episode,
    validate_override,
)

R = 300  # episodes per scenario per arm
SEED_BASE = 1000
N_BOOT = 5000


def seed_for(scenario_idx: int, episode_idx: int) -> int:
    return SEED_BASE + scenario_idx * 100_000 + episode_idx


def main() -> dict:
    arms = ("freetext", "freetext_repeat", "sai")
    results: dict[str, list] = {a: [] for a in arms}
    for si, sc in enumerate(SCENARIOS):
        for ei in range(R):
            s = seed_for(si, ei)
            for arm in arms:
                results[arm].append(run_episode(sc, arm, s))

    n_ep = len(SCENARIOS) * R

    def curve(arm):
        import numpy as np
        arr = np.array([r.compliance for r in results[arm]], dtype=float)
        return [round(float(v), 4) for v in arr.mean(axis=0)]

    curves = {a: curve(a) for a in arms}

    # Gate 2: erosion gap on the free-text arm, paired per episode.
    c0 = [1.0 if r.compliance[0] else 0.0 for r in results["freetext"]]
    c9 = [1.0 if r.compliance[9] else 0.0 for r in results["freetext"]]
    gap_mean, gap_lo, gap_hi = paired_bootstrap_ci(c9, c0, n_boot=N_BOOT, seed=7)
    gate2_pass = gap_mean >= 0.20 and gap_lo > 0

    # Gate 3: SAI vs free-text margin, paired per episode.
    m_base = episode_means(results["freetext"])
    m_rep = episode_means(results["freetext_repeat"])
    m_sai = episode_means(results["sai"])
    marg_mean, marg_lo, marg_hi = paired_bootstrap_ci(m_base, m_sai, n_boot=N_BOOT, seed=8)
    gate3_pass = marg_mean >= 0.15 and marg_lo > 0

    # Overhead: total prompt tokens per episode.
    tok = {a: round(float(__import__("numpy").mean(episode_tokens(results[a]))), 1)
           for a in arms}

    # Schema validation spot-check: 24 well-formed accepted, 50 malformed rejected.
    good = sum(validate_override(sc.typed_payload)[0] for sc in SCENARIOS)
    bad_payloads = [
        {}, {"class": "constraint"}, {"class": "ban", "target": "x", "directive": "y" * 20},
        {"class": "constraint", "target": "Bad Target!", "directive": "do the thing please"},
        {"class": "priority", "target": "ok:target", "directive": "short"},
        {"class": "strategy", "target": "ok:target", "directive": "z" * 400},
        "not-a-dict", None, 42,
    ] * 6
    bad_payloads = bad_payloads[:50]
    bad_rejected = sum(not validate_override(p)[0] for p in bad_payloads)

    summary = {
        "episodes_per_arm": n_ep,
        "scenarios": len(SCENARIOS),
        "turns": 10,
        "curves": curves,
        "gate2_erosion": {"mean": round(gap_mean, 4), "ci95": [round(gap_lo, 4), round(gap_hi, 4)],
                          "bar": ">=0.20 and CI excludes 0", "pass": bool(gate2_pass)},
        "gate3_margin": {"mean": round(marg_mean, 4), "ci95": [round(marg_lo, 4), round(marg_hi, 4)],
                         "bar": ">=0.15 and CI excludes 0", "pass": bool(gate3_pass)},
        "mean_episode_compliance": {
            "freetext": round(float(__import__("numpy").mean(m_base)), 4),
            "freetext_repeat": round(float(__import__("numpy").mean(m_rep)), 4),
            "sai": round(float(__import__("numpy").mean(m_sai)), 4),
        },
        "mean_prompt_tokens_per_episode": tok,
        "schema_validation": {"wellformed_accepted": f"{good}/{len(SCENARIOS)}",
                              "malformed_rejected": f"{bad_rejected}/{len(bad_payloads)}"},
        "honesty": ("deterministic policy simulator; no LLMs involved; "
                    "distractor schedule and decay fixed a priori"),
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
