# Worker brief

Use this file with a dispatched task brief. Work within that brief's scope and
return evidence for review.

## Before editing

1. Create a worktree at the ref named by the task:
   `git worktree add ../BBall-worker-<task> <ref>`.
2. Read `CLAUDE.md`, `ISSUES.md`, the task brief, and the relevant parts of
   `METHODOLOGY.md` and `CONTEXT.md`.
3. Set `OMP_NUM_THREADS=6` for XGBoost-heavy work when other sessions share
   the machine.

Do not change the shared checkout's branch or HEAD.

## Scope limits

- Do not commit to `master`, create version tags, or assign a version number.
- Edit project documentation or the acceptance rules in
  `src/model/evaluate_suite.py` only when the task assigns that work.
- Preserve `outputs/models/evaluation_suite.json` and `oof_reference.csv`
  when an experiment needs more history than the automatic `_prev` copy.
- Use cached data when available. Apply the configured delay to live requests
  and retain valid cached data after a failed fetch.
- Extend `contract_structure_v2.csv` with
  `scripts/extend_contract_structure.py`. Preserve its historical rows.
- Keep temporary experiment scripts outside the repository.

Stop and request review before changing the target definition, evaluation
protocol, or filter-chain membership.

## Required evaluation

Use identical folds, seeds, and rows for reference and candidate.

For a feature candidate, report these gates:

1. Selection: paired `Delta Sel >= +0.002` and `t > 2`.
2. Source control: compare season-aligned data with a season-only arm; compare
   collection-aligned missingness with a coverage-indicator arm.
3. Forecasting: rolling-origin 2024-2026 must move with the pooled gain.
4. Segments: no fixed segment may increase absolute bias by more than $0.3M.

Test the exact missing-value handling and transformations proposed for
production.

Additional rules:

- If the row set changes, compare common rows with a fixed player-to-fold map
  and report the membership change.
- Bin calibration by prediction. Keep segment membership fixed across models.
- Keep `is_confirmation` out of decisions, features, and fill statistics.
- Evaluate each Grabit censor side on its affected zone.
- Run `scripts/check_caps.py` after changing `CAP_BY_SEASON`.
- For `_compute_max_eligible` or `_compute_floor`, verify that observations
  above the computed ceiling remain uncensored. Preserve
  `early_supermax.csv`.

## Deliverable

Return one Markdown report with:

1. Change and rationale, at most two paragraphs.
2. Reference and candidate A1, A2, and B1 metrics.
3. Paired selection delta, standard error, t-statistic, per-fold deltas, and
   control-arm results.
4. Pass/fail for each applicable gate.
5. Suspicious results and source disagreements.
6. Files changed and the branch or patch location.
7. A proposed commit message without a version number.
8. New `ISSUES.md` entries for defects outside the assigned scope.

Escalate when data sources disagree or when a passing result has no clear
mechanism.
