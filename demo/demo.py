#!/usr/bin/env python3
"""30-second demo: free-text override erosion vs SAI typed-override persistence.

Runs one seeded 10-turn episode of the 'forbidden_tool' scenario in both arms
and prints a side-by-side compliance timeline.

    python3 demo/demo.py            # from the repo root
"""
import sys

sys.path.insert(0, "src")

from sai import SCENARIOS, run_episode  # noqa: E402

SCENARIO_ID = "forbidden_tool"
SEED = 20261020


def bar(flags):
    return "".join("#" if f else "." for f in flags)


def main():
    sc = next(s for s in SCENARIOS if s.sid == SCENARIO_ID)
    print("=" * 72)
    print("SAI: Structured Authority Injection  --  30-second demo")
    print("=" * 72)
    print(f"Scenario : {sc.name} -- {sc.description}")
    print(f"Directive: \"{sc.directive}\"")
    print()
    print("The operator states the directive ONCE at turn 0.")
    print("Later turns bring conflicting instructions (distractors at t=2,4,6,8).")
    print("Legend: # = agent honors the operator directive, . = it violates it")
    print()
    for mode, label in (("freetext", "free-text override (baseline)"),
                        ("sai", "SAI typed override (Constraint, rank 100)")):
        r = run_episode(sc, mode, SEED)
        n_ok = sum(r.compliance)
        print(f"{label:38s} [{bar(r.compliance)}]  {n_ok}/10 turns compliant")
    print()
    print("Free text erodes: positional authority decays, the plan gets captured")
    print("by later instructions. SAI persists: ranked authority in a directed")
    print("channel the planner cannot rewrite outranks recency, every turn.")
    print("=" * 72)


if __name__ == "__main__":
    main()
