"""Paired evaluation of the previous-waiver feature and coverage control."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config import OUTPUTS_DIR
from src.model.evaluate_suite import (
    DEFAULT_SEEDS,
    baseline_ladder,
    load_evaluation_frame,
    make_champion_fitter,
    paired_delta,
    run_suite,
)
from src.model.route_mixture import attach_clf_features
from src.model.train import FEATURE_COLS


ARMS = {
    "incumbent": [],
    "waiver": ["is_waived"],
    "coverage": ["is_waived_known"],
    "waiver_plus_coverage": ["is_waived", "is_waived_known"],
}


def _segment_guard(incumbent: dict, challenger: dict) -> dict:
    """Largest signing-mechanism growth in absolute bias."""
    inc = incumbent["C2_by_signing_mechanism"]
    new = challenger["C2_by_signing_mechanism"]
    rows = {}
    for name in sorted(set(inc) & set(new)):
        growth = abs(new[name]["bias_m"]) - abs(inc[name]["bias_m"])
        rows[name] = {
            "n": new[name]["n"],
            "old_bias_m": inc[name]["bias_m"],
            "new_bias_m": new[name]["bias_m"],
            "delta_abs_bias_m": growth,
            "old_mae_m": inc[name]["mae_m"],
            "new_mae_m": new[name]["mae_m"],
        }
    worst = max(rows, key=lambda name: rows[name]["delta_abs_bias_m"])
    return {"worst_segment": worst, "worst_growth_m": rows[worst]["delta_abs_bias_m"],
            "segments": rows}


def main() -> None:
    df, _ = load_evaluation_frame()
    if "is_waived" not in df or "is_waived_known" not in df:
        raise SystemExit("rebuild training_data_v2.csv with waiver columns first")
    df, full_clf = attach_clf_features(df)
    extra_clf = [name for name in full_clf if name not in FEATURE_COLS]
    base = [name for name in FEATURE_COLS if name != "is_waived"]

    known = int(df["is_waived_known"].sum())
    positive = int(df["is_waived"].fillna(0).sum())
    print(
        f"evaluation rows {len(df)}, source-known {known} "
        f"({known / len(df):.1%}), waiver positives {positive}"
    )

    ladder = baseline_ladder(df, base, DEFAULT_SEEDS)
    results = {}
    for name, additions in ARMS.items():
        features = base + additions
        clf_features = features + extra_clf
        print(f"\nARM {name}: {len(features)} regression features")
        results[name] = run_suite(
            df, features, make_champion_fitter(clf_features),
            name, seeds=DEFAULT_SEEDS, ladder=ladder,
        )

    incumbent = results["incumbent"]
    payload = {
        "n": len(df),
        "known_n": known,
        "positive_n": positive,
        "seeds": list(DEFAULT_SEEDS),
        "arms": {},
    }
    for name, result in results.items():
        metrics = result.metrics
        arm = {"metrics": metrics}
        if name != "incumbent":
            arm["paired_pooled"] = paired_delta(
                incumbent.fold_r2, result.fold_r2
            )
            arm["paired_selection"] = paired_delta(
                incumbent.fold_r2_sel, result.fold_r2_sel
            )
            arm["segment_guard"] = _segment_guard(
                incumbent.metrics, metrics
            )
            arm["calibration_excess"] = (
                abs(metrics["C1_calibration_slope"] - 1)
                - abs(incumbent.metrics["C1_calibration_slope"] - 1)
            )
            arm["forward_delta"] = (
                metrics["B1_forward_r2"]
                - incumbent.metrics["B1_forward_r2"]
            )
        payload["arms"][name] = arm

    print("\n" + "=" * 82)
    print("WAIVER FEATURE PAIRED SCORECARD")
    print("=" * 82)
    old = incumbent.metrics
    for name in ("waiver", "coverage", "waiver_plus_coverage"):
        arm = payload["arms"][name]
        metrics = arm["metrics"]
        delta = arm["paired_selection"]
        guard = arm["segment_guard"]
        print(
            f"{name:24s} A1 {metrics['A1_cv_r2']:.4f} "
            f"deltaSel {delta['delta']:+.5f} +/- {delta['se']:.5f} "
            f"t={delta['t']:+.2f}  "
            f"A2 {metrics['A2_cv_r2_2024_26'] - old['A2_cv_r2_2024_26']:+.4f} "
            f"B1 {arm['forward_delta']:+.4f}  "
            f"C1 excess {arm['calibration_excess']:+.4f}  "
            f"worst C2 {guard['worst_segment']} "
            f"{guard['worst_growth_m']:+.3f}M"
        )
        print(f"  folds {delta['per_fold']}")

    out = OUTPUTS_DIR / "models" / "waiver_feature_evaluation.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
