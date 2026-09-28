"""Paired evaluation of ex-ante Spotrac injury-history scores.

The fixed score ladder separates the requested weighting choices:

* ``time_1y``: log games missed with a one-year half-life.
* ``time_2y``: the same impact with a two-year half-life.
* ``typed``: adds fixed injury-family severity weights.
* ``recurrent``: also raises repeated same-family events by up to 75%.

Use ``--screen`` for Layer A only.  A formal run without it evaluates the
pre-selected recurrent arm and both source controls through A1/A2/B1/C/D.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from config import OUTPUTS_DIR
from src.features.injury_history import attach_injury_history
from src.model.evaluate_suite import (
    DEFAULT_SEEDS,
    N_SPLITS,
    TARGET,
    _compute_kf_nested,
    _compute_kf_forward,
    baseline_ladder,
    load_evaluation_frame,
    make_kf_champion_fitter,
    paired_delta,
    prepare_kf_context,
    layer_a,
    layer_b,
    layer_c,
    layer_d,
)
from src.model.route_mixture import attach_clf_features
from src.model import route_mixture as rm
from src.model.stages import compose
from src.model.train import FEATURE_COLS


SCORE_ARMS = {
    "incumbent": [],
    "time_1y": ["injury_score_1y"],
    "time_2y": ["injury_score_2y"],
    "typed": ["injury_score_typed"],
    "recurrent": ["injury_score_recurrent"],
    "games_observed": ["injury_games_observed"],
    "event_count": ["injury_event_count"],
    "coverage": ["injury_history_known"],
    "season_control": ["injury_season_control"],
    "recurrent_plus_coverage": [
        "injury_score_recurrent", "injury_history_known",
    ],
}
FORMAL_ARMS = (
    "incumbent", "typed", "recurrent", "coverage", "season_control",
)
OUT = OUTPUTS_DIR / "models" / "injury_feature_evaluation.json"


def _frame():
    df, _ = load_evaluation_frame(
        verbose=False, allow_missing_computed=True
    )
    df = attach_injury_history(df)
    # Zero means no observable health burden.  Unknown source coverage is also
    # zero in the score arm and is tested separately by the coverage arm.
    score_cols = sorted({c for cols in SCORE_ARMS.values() for c in cols})
    df["injury_season_control"] = df["season"].astype(float)
    df[score_cols] = df[score_cols].fillna(0.0)
    df, full_clf = attach_clf_features(df)
    extra_clf = [name for name in full_clf if name not in FEATURE_COLS]
    df = df.reset_index(drop=True)
    kf_ctx = prepare_kf_context(df, full_clf, verbose=False)
    return df, full_clf, kf_ctx


def _segment_guard(incumbent: dict, challenger: dict) -> dict:
    old = incumbent["C2_by_signing_mechanism"]
    new = challenger["C2_by_signing_mechanism"]
    growth = {
        name: abs(new[name]["bias_m"]) - abs(old[name]["bias_m"])
        for name in sorted(set(old) & set(new))
    }
    worst = max(growth, key=growth.get)
    return {"worst_segment": worst, "worst_growth_m": growth[worst],
            "all_growth_m": growth}


def _fitter(clf_features, kf_ctx):
    # Injury burden enters the regression only.  The route classifier stays on
    # its frozen feature list, so the candidate cannot win by changing the
    # probability-based max intervention instead of market-value estimation.
    return make_kf_champion_fitter(kf_ctx, clf_features)


def _shared_oof(df, clf_features, kf_ctx, seeds, arm_names):
    """Layer-A predictions with one shared nested-KF pass per fold.

    KF measurements and route probabilities do not depend on which injury
    score enters the final regression, so computing them once makes the arms
    paired more tightly and avoids repeating the expensive inner fits.
    """
    base = list(FEATURE_COLS)
    y = df[TARGET].values
    selection = ~df["is_confirmation"].values
    folds = list(GroupKFold(n_splits=N_SPLITS).split(
        df, y, df["player_name_norm"].values
    ))
    results = {
        name: {
            "oof": np.zeros(len(df)),
            "fold": np.zeros((len(folds), len(seeds))),
            "fold_sel": np.zeros((len(folds), len(seeds))),
        }
        for name in arm_names
    }
    for si, seed in enumerate(seeds):
        for fi, (tr, va) in enumerate(folds):
            train, test = df.iloc[tr], df.iloc[va]
            tr_aug, te_aug = _compute_kf_nested(
                kf_ctx, train, test, clf_features, seed
            )
            route = rm.train_route_classifier(tr_aug, clf_features, seed)
            p_max = rm.route_proba(route, te_aug, clf_features)[:, rm.MAX_IDX]
            for name in arm_names:
                additions = SCORE_ARMS[name]
                features = base + additions
                latent, lo, hi = rm.grabit_latent(
                    tr_aug, te_aug, features, seed
                )
                pred = compose(
                    latent, lo=lo, hi=hi, p_max=p_max,
                    is_extension=test["is_extension"].values,
                    ext_cap_pct=test["ext_cap_pct"].values,
                )
                results[name]["fold"][fi, si] = r2_score(y[va], pred)
                keep = selection[va]
                results[name]["fold_sel"][fi, si] = r2_score(
                    y[va][keep], pred[keep]
                )
                results[name]["oof"][va] += pred
            print(
                f"  seed {seed} fold {fi + 1}/{len(folds)} complete",
                flush=True,
            )
    for rec in results.values():
        rec["oof"] /= len(seeds)
    return results


def screen(df, clf_features, kf_ctx, seeds) -> dict:
    """Layer-A paired screen of the fixed weighting ladder."""
    results = _shared_oof(
        df, clf_features, kf_ctx, seeds, tuple(SCORE_ARMS)
    )

    ref = results["incumbent"]
    payload = {"n": len(df), "seeds": list(seeds), "arms": {}}
    for name, rec in results.items():
        if name == "incumbent":
            continue
        pooled = paired_delta(ref["fold"], rec["fold"])
        selection = paired_delta(ref["fold_sel"], rec["fold_sel"])
        payload["arms"][name] = {
            "paired_pooled": pooled, "paired_selection": selection,
        }
        print(
            f"{name:26s} dSel {selection['delta']:+.5f} "
            f"+/- {selection['se']:.5f} t={selection['t']:+.2f} "
            f"folds={selection['per_fold']}"
        )
    return payload


def _shared_forward(df, clf_features, kf_ctx, seeds, arm_names):
    """Rolling-origin predictions with one KF computation per origin."""
    base = list(FEATURE_COLS)
    season = df["season"].values
    out = {name: np.full(len(df), np.nan) for name in arm_names}
    for origin in (2024, 2025, 2026):
        test_mask = season == origin
        train_mask = season < origin
        if test_mask.sum() < 10 or train_mask.sum() < 200:
            continue
        kf_all = _compute_kf_forward(
            df, kf_ctx, train_mask, clf_features, seeds
        )
        fill = float(np.nanmedian(kf_all[train_mask]))
        train_aug = df[train_mask].copy()
        test_aug = df[test_mask].copy()
        train_aug["kf_market_value"] = np.where(
            np.isfinite(kf_all[train_mask]), kf_all[train_mask], fill
        )
        test_aug["kf_market_value"] = np.where(
            np.isfinite(kf_all[test_mask]), kf_all[test_mask], fill
        )
        acc = {name: np.zeros(int(test_mask.sum())) for name in arm_names}
        for seed in seeds:
            route = rm.train_route_classifier(train_aug, clf_features, seed)
            p_max = rm.route_proba(
                route, test_aug, clf_features
            )[:, rm.MAX_IDX]
            for name in arm_names:
                features = base + SCORE_ARMS[name]
                latent, lo, hi = rm.grabit_latent(
                    train_aug, test_aug, features, seed
                )
                acc[name] += compose(
                    latent, lo=lo, hi=hi, p_max=p_max,
                    is_extension=test_aug["is_extension"].values,
                    ext_cap_pct=test_aug["ext_cap_pct"].values,
                )
        for name in arm_names:
            out[name][test_mask] = acc[name] / len(seeds)
        print(f"  forward origin {origin} complete", flush=True)
    return out


def formal(df, clf_features, kf_ctx, seeds) -> dict:
    """Full suite for the pre-selected score and source controls."""
    base = list(FEATURE_COLS)
    ladder = baseline_ladder(df, base, seeds)
    results = _shared_oof(df, clf_features, kf_ctx, seeds, FORMAL_ARMS)
    forward = _shared_forward(df, clf_features, kf_ctx, seeds, FORMAL_ARMS)
    for name in FORMAL_ARMS:
        rec = results[name]
        metrics = {}
        metrics.update(layer_a(df, rec["oof"], rec["fold"]))
        metrics.update(layer_b(df, forward[name]))
        metrics.update(layer_c(df, rec["oof"]))
        metrics.update(layer_d(
            df, base + SCORE_ARMS[name], rec["oof"], seeds, ladder=ladder
        ))
        rec["metrics"] = metrics

    ref = results["incumbent"]
    payload = {
        "n": len(df),
        "known_n": int(df["injury_history_known"].sum()),
        "positive_n": int(df["injury_score_recurrent"].gt(0).sum()),
        "seeds": list(seeds),
        "arms": {},
    }
    for name, result in results.items():
        arm = {"metrics": result["metrics"]}
        if name != "incumbent":
            arm["paired_pooled"] = paired_delta(
                ref["fold"], result["fold"]
            )
            arm["paired_selection"] = paired_delta(
                ref["fold_sel"], result["fold_sel"]
            )
            arm["forward_delta"] = (
                result["metrics"]["B1_forward_r2"]
                - ref["metrics"]["B1_forward_r2"]
            )
            arm["segment_guard"] = _segment_guard(
                ref["metrics"], result["metrics"]
            )
        payload["arms"][name] = arm

    print("\nFORMAL INJURY SCORECARD")
    for name in FORMAL_ARMS[1:]:
        arm = payload["arms"][name]
        metric = arm["metrics"]
        delta = arm["paired_selection"]
        guard = arm["segment_guard"]
        print(
            f"{name:26s} A1={metric['A1_cv_r2']:.4f} "
            f"dSel={delta['delta']:+.5f} t={delta['t']:+.2f} "
            f"A2={metric['A2_cv_r2_2024_26']:.4f} "
            f"B1d={arm['forward_delta']:+.4f} "
            f"worstC2={guard['worst_segment']} "
            f"{guard['worst_growth_m']:+.3f}M"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--seeds", type=int, default=10)
    args = parser.parse_args()
    seeds = tuple(DEFAULT_SEEDS[: args.seeds])
    df, clf_features, kf_ctx = _frame()
    print(
        f"evaluation rows {len(df)}, signing-date known "
        f"{int(df['injury_history_known'].sum())}, positive burden "
        f"{int(df['injury_score_recurrent'].gt(0).sum())}"
    )
    lillard = df[
        (df["player_name_norm"] == "damian lillard")
        & (df["season"] == 2025)
    ]
    if not lillard.empty:
        print("Damian Lillard 2025 ex-ante injury fields:")
        print(lillard[[
            "injury_cutoff_date", "injury_score_recurrent",
            "injury_event_count", "injury_games_observed",
        ]].to_string(index=False))
    payload = screen(df, clf_features, kf_ctx, seeds) if args.screen else formal(
        df, clf_features, kf_ctx, seeds
    )
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
