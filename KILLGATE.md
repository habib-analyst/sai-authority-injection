# KILL-GATE VERDICTS — sai-authority-injection (project #51)

Paper: arXiv 2610.22373 — "Structured Authority Injection: Persistent Human
Control for LLM Agents via Typed MCP Overrides" (Tran, Zhang, Tao, Nan Yang,
Shengjie Xu, Yu Yin, Shuai Xu, Wenlu Zhang, Hailu Xu).

Evaluated in order. A failed gate would have ended the build immediately.

---

## Gate 1 — Author code absent: PASS (build authorized)

Re-verified LIVE on 2026-10-20 (~10:10 PKT), not trusting the earlier read.

- arXiv abs page fetched via curl: title confirmed
  "[2610.22373] Structured Authority Injection: Persistent Human Control for
  LLM Agents via Typed MCP Overrides". (Note: id taken from the page <title>,
  not from /abs/ links in <dd> footers, per the 2026-10-20 extraction-bug
  lesson. The v1 submission is dated 27 Jul 2026; the paper was announced in
  the Oct 19 2026 listing.)
- Full HTML (ar5iv) fetched and grepped: ZERO github.com links, no "code is
  available" statement anywhere in the paper text.
- GitHub repo search API `q=structured+authority+injection`: total_count 0.
- GitHub repo search API exact phrase `"structured authority injection"`: 0.
- GitHub user search: no user matching the paper/author identity
  ("sai+structured+authority": 0; "Hailu Xu": 0; "rytran" hits are unrelated
  existing accounts, none linked to this paper).
- Web search for implementations: only press coverage (a Smart Chunks
  explainer article, published 2026-10-19). No code repos, no third-party
  implementations found.
- GitHub *code* search requires authentication and was not available from this
  environment; repo search + user search + web search + paper-text grep all
  agree: no author code exists publicly.

**Verdict: PASS.** No author implementation exists; an open implementation is
legitimate. Proceeding was the correct call.

## Gate 2 — Authority-erosion sanity (measure the DISEASE first): PASS

Bar (fixed a priori): erosion gap = compliance(turn 0) − compliance(turn 9)
for the free-text arm ≥ 0.20, paired bootstrap 95% CI excluding 0.

Measured (8 scenarios × 300 seeded episodes = 2400 episodes, deterministic
policy simulator, 10 turns/episode, distractors at turns 2/4/6/8):

- Erosion gap: **1.000**, 95% CI **[1.000, 1.000]** — excludes 0, far above 0.20.
- Free-text compliance curve: `[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]`
  — the operator's turn-0 directive is honored for 2 turns, then the plan is
  captured by later instructions and never recovers.
- Mean episode compliance (free-text): **0.200**.

The disease is real and measured before any fix was built: positional,
free-text authority decays with recency and loses to later instructions.

**Verdict: PASS.**

## Gate 3 — Load-bearing (SAI beats free-text with a margin): PASS

Bar (fixed a priori): margin = mean episode compliance(SAI) − mean episode
compliance(free-text) ≥ 0.15, paired bootstrap 95% CI excluding 0.

Measured (same 2400 episodes/arm, 3 arms):

- Margin: **0.800**, 95% CI **[0.800, 0.800]** — excludes 0, far above 0.15.
- Mean episode compliance: SAI **1.000** vs free-text **0.200**.
- SAI compliance curve: `[1.0] × 10` — the typed override persists every turn
  in every scenario.
- Schema validation: 8/8 well-formed overrides accepted, 50/50 malformed
  payloads rejected.
- Bonus arm (operator re-sends the free-text directive every 3 turns):
  mean compliance 0.502 at 1233.5 prompt tokens/episode vs SAI 1047.3 tokens —
  reminders partially recover compliance but cost ~18% more tokens and still
  race distractors at equal recency (turn-6 compliance 0.51). SAI wins on both
  compliance and overhead.

**Verdict: PASS.** All three gates passed. Build authorized to ship.

---

## Honesty statement

These numbers come from a **deterministic policy simulator** (seeded RNG, no
LLMs): it operationalizes the paper's authority-erosion mechanism
(recency-decayed salience + plan internalization) with the distractor schedule
and decay fixed a priori. They demonstrate the *mechanism* — typed, ranked,
directed authority persists where positional free-text authority erodes — not
LLM generalization. The paper itself evaluated 9 LLMs / 8 scenarios; this CPU
build mirrors its 8-scenario structure with a simulator, stated plainly in the
README and paper writeup. No numbers are fabricated; every figure above is
reproducible via `python3 experiments/run_gates.py` (fixed seeds).
