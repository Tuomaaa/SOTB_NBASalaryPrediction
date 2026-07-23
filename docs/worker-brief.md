# Worker brief — read this before touching anything

You are a worker agent on the NBA free-agent valuation model. You execute ONE
task, defined by a task brief handed to you alongside this file, and deliver an
evidence bundle. An architect session reviews your evidence and makes every
adoption decision. You produce proof; you do not rule on it.

## Your boundaries

You may read everything and run anything read-only. You may create and edit
files inside your task's scope and your scratchpad. You must NOT:

- commit to `master` or create git tags — deliver a branch or patch; the
  architect lands it and assigns the `vN.Mx` version if numbers move
- edit `CLAUDE.md`, `METHODOLOGY.md`, `VERSION_HISTORY.md`, `CONTEXT.md`,
  `PROJECT_BRIEF.md` (docs agent's lane) or the accept/reject rules in
  `src/model/evaluate_suite.py` (architect's lane)
- overwrite `outputs/models/evaluation_suite.json` / `oof_reference.csv`
  carelessly — the suite auto-archives one `_prev` generation; if you need
  more history, copy aside first
- re-scrape anything already cached, or scrape without the 3s+ rate limits in
  `config.py` — a full BBRef team refresh takes ~25 min and one team will 403;
  a failed fetch must degrade to stale data, never to missing data
- regenerate `data/processed/contract_structure_v2.csv` from scratch — the
  original generator is lost and reconstructions reach only 88% fidelity;
  `scripts/extend_contract_structure.py` extends it incrementally and
  hard-fails if a pre-existing row would move

## Reading order

1. `CLAUDE.md` — conventions and the load-bearing warnings
2. `ISSUES.md` — your problem may already be written up; append what you find
   and do not fix (entries must work without your conversation)
3. Your task brief — scope, deliverable, and the tag you are pinned to
4. `METHODOLOGY.md` sections relevant to your task; `CONTEXT.md` for vocabulary

Pin your work to the git tag named in your task brief (`git log --oneline`
and `git tag -n1` orient you). Every published number reproduces from its tag.

## Hard rules (each one was paid for)

**Code**
- Every new entry point under `src/` or `scripts/` starts with the repo-root
  `sys.path.insert` bootstrap or `from config import ...` will not resolve
- `sys.stdout.reconfigure(encoding="utf-8")`, never rebind `sys.stdout` — the
  old wrapper gets GC'd and closes the buffer for importers
- Experiment scripts live in your scratchpad, not the repo

**Evaluation**
- Accept/reject reads ONE thing: the paired delta on the selection pool,
  same folds and seeds as the reference matrices in
  `outputs/models/evaluation_suite.json` (`fold_r2_selection`). Fold sd is
  ~0.046, seed sd ~0.001 — unpaired comparisons throw away the power
- R² is not comparable across different row sets (D1). If your change alters
  the row count, report common-row deltas, never headline-vs-headline
- Calibration and segment tables bin by PREDICTED value, and model-vs-model
  segment comparisons use FIXED row groups — binning by the target or by each
  model's own bands manufactures regression-to-the-mean artifacts
- The confirmation split (`is_confirmation`) is a canary. Never let it into a
  decision metric, a fill statistic, or a feature derivation

**Feature candidates run four gates, in order**
1. paired ΔSel ≥ +0.002 with t > 2
2. source control: if values or missingness align with seasons, beat a pure
   season-dummy arm (an `is2019` flag with zero content scored +0.0032 here);
   if missingness aligns with our own collection process, score as an
   increment over an explicit coverage-indicator arm (that artifact alone
   scored +0.0075)
3. B1 forward veto: rolling-origin 2024-26 must move WITH the pooled gain —
   and know its blind spot: a feature can absorb a season offset on the
   training side without leaking into test origins, which is what gate 2 is for
4. C2: no fixed-row segment's |bias| grows by more than $0.3M
Test the SHIP FORM — the exact NaN handling that would go live. Native-NaN
scored +0.0046 where the median-filled ship form of the same feature scored
+0.0001.

**Model structure**
- Grabit's two censored sides are judged on their zone scorecards (printed by
  the suite), never on the pooled delta — ~30% of rows sit at a CBA bound and
  pooled metrics dilute a targeted intervention ~20x
- Anything touching `_compute_max_eligible` or `_compute_floor`: over-cap rows
  (actual above ceiling) must stay at zero, and the curated
  `early_supermax.csv` exists because a blanket award-window widening
  un-censored two genuine 30% max signings — do not widen windows to cover
  individual cases

**Data**
- Salary caps live ONLY in `config.py`; `cap_pct` is recomputed at load.
  After touching `CAP_BY_SEASON`, run `scripts/check_caps.py`
- The refresh chain is `scripts/refresh_salaries.py` →
  `scripts/extend_contract_structure.py` → `scripts/rebuild_training_data.py`,
  each validating that pre-existing rows survive byte-for-byte

## The evidence bundle you deliver

A single markdown report (plus artifacts) containing:

1. **What changed and why** — two paragraphs maximum
2. **Numbers table** — A1 / A2 / B1 for reference and candidate, the paired
   ΔSel ± SE and t, per-fold deltas, and every control arm you ran
3. **Gate verdicts** — each of the four gates, pass/fail, with the number
4. **Anomalies** — anything suspicious, especially results that look too good;
   a delta above +0.01 is a red flag until its source is explained, not a win
5. **Files touched** and where your branch/patch lives
6. **Proposed commit message** — the architect edits and lands it; do not
   claim a version number, that assignment is the architect's
7. **ISSUES.md additions** for anything real you found and did not fix

## Stop and escalate immediately if

- your task would change the target definition, the filter chain's row count,
  or the evaluation protocol
- you find a result that passes the gates but whose mechanism you cannot
  explain in one sentence
- two instruments disagree (a span says one thing, a salary step another —
  see ISSUES #2's warning about span-only evidence)
- anything requires editing files in another agent's lane

---

## Architect's section (not for workers)

Dispatch: pick a chunky task (≥ half a day), write a task brief naming the
pinned tag, scope, deliverable format, and known traps; hand the worker this
file plus the brief. Review: reproduce headline numbers from the pinned tag,
rerun any control the evidence lacks, then land with `vN.Mx` tag or file the
rejection with its numbers. One number-moving workstream at a time; analysis
tasks may run in parallel with it.
