"""Score the KF measurement form on the v6.3.0 champion.

The suite builds `kf_market_value` from measurements that go through the
push and the clip (`evaluate_suite.MEASUREMENT_MODE = "push_clip"`).
`predict.py` and `export_web.py` use the raw measurement-model output
("latent"). Arms, identical in every other respect:

    push_clip   suite form until now (reference)
    clip        clip into [floor, ceiling], no push
    latent      raw output, the production form

    python scripts/eval_kf_measurement.py --seeds 10 --full
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from config import OUTPUTS_DIR
from scripts.eval_pmax_stage1 import named_rows, zones
from src.model import evaluate_suite as suite
from src.model.evaluate_suite import (
    ARM_CHAMPION, DEFAULT_SEEDS, abs_bias_growth, attach_clf_features,
    baseline_ladder, layer_a, layer_c, load_evaluation_frame,
    make_kf_stage_arms_fitter, oof_groupkfold_signing, paired_delta,
    prepare_kf_context, run_suite_arms_kf,
)
from src.model.train import FEATURE_COLS

OUT = OUTPUTS_DIR / "models"
MODES = ["push_clip", "clip", "latent"]


def main() -> None:
    """Score each measurement mode and report paired deltas against push_clip."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--modes", nargs="+", default=MODES)
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])
    modes = ["push_clip"] + [m for m in args.modes if m != "push_clip"]

    df, base_features = load_evaluation_frame(allow_missing_computed=True)
    df, clf_features = attach_clf_features(df)
    features = list(FEATURE_COLS)
    ladder = baseline_ladder(df, base_features, seeds) if args.full else None

    results = {}
    for mode in modes:
        suite.MEASUREMENT_MODE = mode
        print(f"\nMODE {mode}", flush=True)
        kf_ctx = prepare_kf_context(df, base_features)
        fitter = make_kf_stage_arms_fitter(kf_ctx, clf_features)
        if args.full:
            res, _ = run_suite_arms_kf(df, features, fitter, kf_ctx,
                                       clf_features, seeds=seeds,
                                       ladder=ladder)
            r = res[ARM_CHAMPION]
            oof, sel, m = r.oof, r.fold_r2_sel, dict(r.metrics)
        else:
            store, _ = oof_groupkfold_signing(df, features, fitter, seeds)
            oof, fold_r2, sel = store[ARM_CHAMPION]
            m = {**layer_a(df, oof, fold_r2), **layer_c(df, oof)}
        results[mode] = {"oof": oof, "sel": sel, "m": m}
    suite.MEASUREMENT_MODE = "push_clip"

    ref = results["push_clip"]
    report = {}
    for mode in modes:
        m = results[mode]["m"]
        by = m.get("B1_by_origin", {})
        e = {"A1": m["A1_cv_r2"], "A2": m["A2_cv_r2_2024_26"],
             "B1": m.get("B1_forward_r2"),
             "B1_2026": by.get("2026", {}).get("r2"),
             "C1_slope": m["C1_calibration_slope"],
             "zones": zones(df, results[mode]["oof"]),
             "named_m": named_rows(df, results[mode]["oof"])}
        if mode != "push_clip":
            pdl = paired_delta(ref["sel"], results[mode]["sel"])
            a, c = ref["m"]["C2_by_signing_mechanism"], m["C2_by_signing_mechanism"]
            growth = {k: abs_bias_growth(c[k]["bias_m"], a[k]["bias_m"])
                      for k in set(a) & set(c)}
            worst = max(growth, key=growth.get)
            e.update({"dSel": pdl["delta"], "se": pdl["se"], "t": pdl["t"],
                      "per_fold": [float(x) for x in pdl["per_fold"]],
                      "C2_worst": worst, "C2_growth_m": float(growth[worst])})
        report[mode] = e
        b1 = "" if e["B1"] is None else f"  B1 {e['B1']:.4f}"
        b26 = "" if e["B1_2026"] is None else f"  B1-2026 {e['B1_2026']:.4f}"
        print(f"\n{mode}: A1 {e['A1']:.4f}  A2 {e['A2']:.4f}{b1}{b26}  "
              f"slope {e['C1_slope']:.3f}")
        if mode != "push_clip":
            print(f"  dSel {e['dSel']:+.5f} se {e['se']:.5f} t {e['t']:+.2f}  "
                  f"per fold {e['per_fold']}  C2 worst {e['C2_worst']} "
                  f"{e['C2_growth_m']:+.3f}M")
        print(f"  zones {json.dumps(e['zones'])}")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = "full" if args.full else "screen"
    (OUT / f"kf_measurement_{tag}.json").write_text(
        json.dumps({"seeds": list(seeds), "modes": report}, indent=2),
        encoding="utf-8")
    pd.DataFrame({"player": df["player_name_norm"], "season": df["season"],
                  "actual": df["cap_pct"],
                  **{m: results[m]["oof"] for m in modes}}).to_csv(
        OUT / f"kf_measurement_{tag}_oof.csv", index=False)
    print(f"\nwrote {OUT / f'kf_measurement_{tag}.json'}")


if __name__ == "__main__":
    main()
