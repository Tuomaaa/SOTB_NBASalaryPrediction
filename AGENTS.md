# AGENTS.md — NBA Free Agent Valuation Model

The full project instructions live in **[CLAUDE.md](CLAUDE.md)** — architecture,
data sources, feature set, modeling strategy, and coding conventions. Read that
file first. This one exists so that agents which look for `AGENTS.md` by name
find their way there.

## Quick orientation

- **Goal**: predict NBA player market value as `cap_pct` (annual salary / salary
  cap in the signing year), using only publicly available data. An open-source,
  reproducible answer to Hollinger's BORD$.
- **Current model (v7.0x)**: two-stage Grabit pipeline — XGBoost trained with a
  censored-normal loss so CBA-capped max contracts are treated as right-censored
  observations, then `final_pred = min(latent_value, max_eligible_pct)`.
- **Entry point**: `python src/model/train.py` trains and evaluates from raw data.
  Every entry point under `src/` and `scripts/` inserts the repo root into
  `sys.path` at the top of the file — keep that bootstrap when adding new ones,
  or `from config import ...` will not resolve.

## Further reading

| File | Contents |
|------|----------|
| [CLAUDE.md](CLAUDE.md) | Project instructions and coding conventions — **the source of truth** |
| [METHODOLOGY.md](METHODOLOGY.md) | Feature definitions, model math, ablation results, known limitations |
| [VERSION_HISTORY.md](VERSION_HISTORY.md) | v1.0 → v7.0x with CV R² at each step, including the leaked-feature era and why it was reverted |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md) | Outward-facing summary of the research contribution |

Do not add project instructions to this file — put them in CLAUDE.md so there is
only one copy to keep current.
