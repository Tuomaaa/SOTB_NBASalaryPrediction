# NBA Free Agent Valuation Model

This project predicts NBA free-agent market value as **Cap Percentage**
(`cap_pct`, annual salary divided by that season's cap) from public data.
Published results must remain reproducible and free of target-derived features.

## Read order

- `ISSUES.md`: active defects and checks.
- `docs/QUEUE.md`: current work.
- `METHODOLOGY.md`: features, model math, ablations, and limits.
- `CONTEXT.md`: terms used by code and published prose.
- `VERSION_HISTORY.md`: landed versions and tagged metrics.
- `docs/adr/`: architecture decisions.

## Agent-to-agent documentation

Use these rules for all documents that transfer work between agents.

### Classify the document

An active document changes current work. The active document set has five
standard files:

- `AGENTS.md`: entry point and reading order.
- `CLAUDE.md`: current project rules and invariants.
- `docs/worker-brief.md`: common rules for dispatched tasks.
- `docs/QUEUE.md`: active work and completion conditions.
- `ISSUES.md`: active defects.

A task handoff is active only while `docs/QUEUE.md` depends on it. A completed
brief or RESULT is historical evidence.

Do not rewrite historical evidence during an active-document cleanup. Update it
only to correct a factual error in that evidence.

### Give each document one function

Keep each rule in one source file. Add a link when another document needs that
rule.

Do not copy model rules into `AGENTS.md`. Do not copy defect history into
`docs/QUEUE.md`.

Use these document forms:

- Queue item: action and completion condition.
- Issue: Problem, Reproduce, Fix, and Done.
- Task brief: scope, fixed inputs, required evidence, and stop conditions.
- Active handoff: status, current evidence, open decisions, and next action.
- RESULT: change, evidence, decision, and artifact paths.

### Keep only action-changing information

Keep information that does at least one of these jobs:

- Changes the next action.
- Defines a scope limit or invariant.
- Prevents a known failure.
- Gives a reproduction command.
- Defines an acceptance or completion condition.
- Identifies the authoritative source or artifact.

Delete these items from active documents:

- Session narrative and debate history.
- Rhetorical warnings without a specific action.
- Old frame counts that do not define a current baseline.
- Repeated explanations that exist in `METHODOLOGY.md`, an ADR, or a RESULT.
- Rejected alternatives that do not constrain current work.
- Internal role names that do not change authority or scope.

Keep a rejected alternative only when repeating it can cause a known defect.
State the failed action and its consequence in one short rule.

### Control terminology

Keep code identifiers and domain terms that change implementation. Define each
project metric at its first active use.

Use one term for one concept. Remove private session terms, metaphors, and
synonyms that add no technical distinction.

### Maintain the documents

When work lands, remove its queue item and fixed issue. Record the result in
`VERSION_HISTORY.md`.

Never reuse an issue number. If code still cites a retired issue, keep one
short entry in the retired-reference index.

Before completion, make sure that links resolve and issue numbers remain in
order. Run `git diff --check`.

## Current pipeline

`src/model/train.py` trains and evaluates the three-stage model:

1. Stage 1 estimates latent value with a two-sided Grabit loss. The waiver
   term and the P(max) term enter as `base_margin`.
2. Stage 2 applies the CBA bounds.
3. Stage 3 applies known signing-route adjustments and legal caps.

```text
latent -> clip[lo, hi] -> signing offset -> mechanism cap
       -> extension cap -> clip[lo, hi]
```

Composition belongs in `src/model/stages.py`.
`scripts/export_web.py` must refit once and export from that fit. It must not
read historical files under `outputs/predictions/`; see ADR 0001.

Training uses Year-1 Contracts. Scoring can include later contract years. Keep
**Signing Residual** (Year-1 model error) distinct from **Contract Surplus**
(team outcome on any contract year).

The 21 production features are defined in `METHODOLOGY.md`.
`kf_market_value` replaced `prev_cap_pct` in v5.2.0. Inference is two-pass
so the Kalman-filtered trajectory exists before the final fit. Review the
rejected-feature table before proposing another feature.

## Model invariants

- Features must be available at inference and must not derive from the target.
  Fit-time censor masks may use the observed target.
- Signing Mechanism may adjust an output. It must not enter `FEATURE_COLS`.
- Stage-3 signing offsets apply to Bird Rights, Cap Space, Early Bird, and
  Non-Bird. `Sign & Trade` maps to Bird Rights in `signing_cat`.
- A legal cap can only lower a prediction. Reapply `[floor, ceiling]` after
  Stage-3 adjustments.
- Salary caps live in `config.py`. Run `scripts/check_caps.py` after changing
  `CAP_BY_SEASON`.
- Store model values as `cap_pct`. Convert to dollars for display.
- Require a paired CV improvement before adding model complexity.
- Preserve the Free-Agent-list veto in the continuation filter. A Spotrac
  contract span alone misclassifies valid new signings.
- Preserve `early_supermax.csv`. A wider award lookback gives some players an
  incorrect supermax ceiling.

## Evaluation

`src/model/evaluate_suite.py` defines separate layers:

| Metric/layer | Purpose |
|---|---|
| A1 | pooled repeated grouped CV selection R-squared (fixed player-to-fold hash, one partition per seed) |
| A2 | A1 scored on 2024-2026 rows |
| B1 | rolling-origin 2024-2026 forecasting |
| C | calibration and fixed-segment bias |
| D | fixed rows, baseline ladder, and confirmation split |

- Make accept/reject decisions from paired fold deltas on selection rows.
- When a filter changes rows, compare common rows. Folds come from a fixed
  player-to-fold hash, so membership changes do not move other players, but
  R-squared values from different row sets are still not comparable.
- Bin calibration by prediction. Keep segment membership fixed across models.
- Evaluate targeted changes on the rows they affect.
- Keep the 15% confirmation split out of selection, feature derivation, and fill
  statistics until the version-bump confirmation run.
- Apply the feature gates in `docs/worker-brief.md`.

## Data and refresh

Sources are Basketball Reference, Spotrac, nbarapm.com, curated tables under
`data/raw/raw_external/`, and salary-cap values in `config.py`. Cache scraped
HTML under `data/raw/html_cache/`.

Current refresh order:

```text
python scripts/refresh_salaries.py
python scripts/extend_contract_structure.py
python scripts/rebuild_training_data.py
python scripts/refresh_spotrac.py --year N
python scripts/eval_stage3_signing.py
```

The final command writes `data/raw/raw_external/signing_offsets.json`, which
`src/model/stages.py` loads. Run it after every rebuild; see ISSUES #48.

`scripts/rebuild_training_data.py` runs:

```text
src/features/build_dataset.py
scripts/build_external_features.py
scripts/phase3.py::build_contract_features
```

- A failed fetch must retain the last valid cached data.
- Wait at least 3 seconds between live requests.
- Extend `data/processed/contract_structure_v2.csv` with
  `scripts/extend_contract_structure.py`. Preserve its historical rows.
- Salaries come from Spotrac cap hits through
  `scripts/build_merged_salaries.py`, with BBRef as the fallback. Read
  VERSION_HISTORY v6.0.0 and ISSUES #28 and #36 before changing
  salary-source logic.

## Code

- Use Python 3.10+ and add docstrings to functions under `src/`.
- Every entry point under `src/` and `scripts/` must insert the repository
  root into `sys.path` before importing `config`.

```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
```

Scripts directly under `scripts/` use `.parent.parent`.

- Configure stdout with `sys.stdout.reconfigure(encoding="utf-8")`. Do not
  replace `sys.stdout`.
- Keep experiments outside production entry points until accepted.

## Finish

Record an unfixed defect in `ISSUES.md` with its problem, reproduction,
proposed fix, and completion check. Delete fixed entries and record landed
changes in `VERSION_HISTORY.md`.

A change that moves published numbers takes the next `PROUD.DEFAULT.SHAME`
version and a git tag containing its headline metrics: PROUD for a structural
advance, DEFAULT for a change that passes the paired gate, SHAME for a
correction adopted on correctness. A SHAME version may also mark a landed
correction or a recorded rejection that moves no published number; its tag
repeats the current headline metrics. `VERSION_HISTORY.md` maps the legacy
`vN.Mx` names used up to v5.3.3.
