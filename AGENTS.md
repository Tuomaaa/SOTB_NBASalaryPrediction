# AGENTS.md — NBA Free Agent Valuation Model

The full project instructions live in **[CLAUDE.md](CLAUDE.md)** — architecture,
data sources, feature set, modeling strategy, and coding conventions. Read that
file first. This one exists so that agents which look for `AGENTS.md` by name
find their way there.

## Quick orientation

- **Goal**: predict NBA player market value as `cap_pct` (annual salary / salary
  cap in the signing year), using only publicly available data. An open-source,
  reproducible answer to Hollinger's BORD$.
- **Current model (v7.8x)**: two-stage Grabit pipeline. Stage 1 predicts value
  under *default parameters* with a two-sided censored-normal loss — max
  contracts are right-censored (the observation floors the latent), veteran
  minimums are left-censored (the league floor props pay above the latent).
  Stage 2 adjusts for *told parameters*, today the two CBA bounds:
  `final_pred = clip(latent, floor_pct, max_eligible_pct)`. Every number-moving
  change takes the next `vN.Mx` and is git-tagged with its headline metrics, so
  any quoted figure reproduces from a checkout.
- **Entry point**: `python src/model/train.py` trains and evaluates from raw data.
  Every entry point under `src/` and `scripts/` inserts the repo root into
  `sys.path` at the top of the file — keep that bootstrap when adding new ones,
  or `from config import ...` will not resolve.
- **Web export**: `python scripts/export_web.py` refits the model and writes the
  portfolio site's data and charts. It reads none of the CSVs in
  `outputs/predictions/` — those span several model versions and disagree with
  each other. See [ADR 0001](docs/adr/0001-single-fit-for-published-numbers.md).

## Further reading

| File | Contents |
|------|----------|
| [CLAUDE.md](CLAUDE.md) | Project instructions and coding conventions — **the source of truth** |
| [ISSUES.md](ISSUES.md) | Known problems nobody has fixed yet — **read before starting, append before stopping** |
| [CONTEXT.md](CONTEXT.md) | Domain glossary — the vocabulary the code and the site both use |
| [docs/adr/](docs/adr/) | Architecture decisions and why the alternatives were rejected |
| [METHODOLOGY.md](METHODOLOGY.md) | Feature definitions, model math, ablation results, known limitations |
| [VERSION_HISTORY.md](VERSION_HISTORY.md) | v1.0 → v7.8x with CV R² at each step, the versioning convention, and why the leaked-feature era was reverted |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Outward-facing summary of the research contribution |

Do not add project instructions to this file — put them in CLAUDE.md so there is
only one copy to keep current.
