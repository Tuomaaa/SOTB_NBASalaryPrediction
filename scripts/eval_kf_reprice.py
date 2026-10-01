"""Evaluate the kf_reprice anchor rule against the incumbent (pre-registered).

The candidate anchors the KF at in-season signings
(`prepare_kf_context(reprice=True)`, see `load_reprice_events`). Both arms
run the full suite with the same folds and seeds. Affected rows are those
whose anchor value or tier changes; with fewer than 20% of rows, the targeted
gate in `docs/worker-brief.md` applies.

    python scripts/eval_kf_reprice.py --seeds 10
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import OUTPUTS_DIR
from scripts.eval_waiver_challengers import c2_worst, targeted_gate
from src.model.evaluate_suite import (
    ARM_CHAMPION, DEFAULT_SEEDS, attach_clf_features, baseline_ladder,
    load_evaluation_frame, make_kf_stage_arms_fitter, paired_delta,
    prepare_kf_context, run_suite_arms_kf,
)
from src.model.train import FEATURE_COLS

OUT = OUTPUTS_DIR / "models"
NAMED = [("russell westbrook", 2023), ("kyle lowry", 2024),
         ("andre drummond", 2021), ("wesley matthews", 2019),
         ("gorgui dieng", 2021), ("thaddeus young", 2022)]


def main() -> None:
    """Run both arms and print the pre-registered gates."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])

    df, base_features = load_evaluation_frame(allow_missing_computed=True)
    df, clf_features = attach_clf_features(df)
    ladder = baseline_ladder(df, base_features, seeds)
    ctx = {"incumbent": prepare_kf_context(df, base_features, reprice=False),
           "kf_reprice": prepare_kf_context(df, base_features, reprice=True)}
    a0 = np.nan_to_num(ctx["incumbent"].anchor, nan=-1.0)
    a1 = np.nan_to_num(ctx["kf_reprice"].anchor, nan=-1.0)
    affected = ((np.abs(a0 - a1) > 1e-12)
                | (ctx["incumbent"].tier != ctx["kf_reprice"].tier))
    share = float(affected.mean())
    print(f"affected rows {int(affected.sum())} ({share:.1%})", flush=True)

    res = {}
    for name, c in ctx.items():
        print(f"\nARM {name}", flush=True)
        fitter = make_kf_stage_arms_fitter(c, clf_features)
        arms, _ = run_suite_arms_kf(df, list(FEATURE_COLS), fitter, c,
                                    clf_features, seeds=seeds, ladder=ladder)
        res[name] = arms[ARM_CHAMPION]

    ref, cand = res["incumbent"], res["kf_reprice"]
    m0, m1 = ref.metrics, cand.metrics
    pdl = paired_delta(ref.fold_r2_sel, cand.fold_r2_sel)
    seg, growth = c2_worst(df, m0, m1)
    c1_gap = (abs(m1["C1_calibration_slope"] - 1)
              - abs(m0["C1_calibration_slope"] - 1))
    tg = targeted_gate(df, ref.oof, cand.oof, affected)
    b1_move = m1["B1_forward_r2"] - m0["B1_forward_r2"]

    y, cap = df["cap_pct"].values, df["cap"].values / 1e6
    sel = ~df["is_confirmation"].astype(bool).values
    d_sel = r2_score(y[sel], cand.oof[sel]) - r2_score(y[sel], ref.oof[sel])
    d_can = (r2_score(y[~sel], cand.oof[~sel])
             - r2_score(y[~sel], ref.oof[~sel]))

    print("\n" + "=" * 70)
    for name, m in (("incumbent", m0), ("kf_reprice", m1)):
        print(f"{name:11s} A1 {m['A1_cv_r2']:.4f}  A2 {m['A2_cv_r2_2024_26']:.4f}"
              f"  B1 {m['B1_forward_r2']:.4f}  slope {m['C1_calibration_slope']:.3f}")
    print(f"paired selection dSel {pdl['delta']:+.5f}  se {pdl['se']:.5f}  "
          f"t {pdl['t']:+.2f}  per fold {pdl['per_fold']}")
    print(f"targeted (affected selection rows): {tg['rows']} rows / "
          f"{tg['players']} players, t {tg['t']:+.2f}, improved "
          f"{tg['share_improved']:.0%}")
    print(f"C2 worst {seg} {growth:+.3f}M   C1 gap {c1_gap:+.4f}   "
          f"B1 move {b1_move:+.4f}")
    if abs(d_can - d_sel) > 0.005:
        print(f"CANARY DIVERGENCE: selection {d_sel:+.4f}, "
              f"confirmation {d_can:+.4f}")
    if share < 0.20:
        ok = (tg["t"] > 2 and pdl["delta"] > 0 and b1_move >= 0
              and growth <= 0.30 and c1_gap <= 0.005)
        print(f"targeted gate: {'PASS' if ok else 'FAIL'}")
    else:
        ok = (pdl["delta"] >= 0.002 and pdl["t"] > 2 and b1_move >= 0
              and growth <= 0.30 and c1_gap <= 0.005)
        print(f"standard gate: {'PASS' if ok else 'FAIL'}")
    e0, e1 = (ref.oof - y) * cap, (cand.oof - y) * cap
    for p, s in NAMED:
        m = ((df["player_name_norm"] == p) & (df["season"] == s)).values
        if m.any():
            i = int(np.flatnonzero(m)[0])
            print(f"  {p} {s}: actual {y[i] * cap[i]:.2f}M  error "
                  f"{e0[i]:+.2f} -> {e1[i]:+.2f}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kf_reprice_eval.json").write_text(json.dumps({
        "seeds": list(seeds), "affected_rows": int(affected.sum()),
        "affected_share": share,
        "incumbent": {k: m0[k] for k in ("A1_cv_r2", "A2_cv_r2_2024_26",
                                          "B1_forward_r2",
                                          "C1_calibration_slope")},
        "kf_reprice": {k: m1[k] for k in ("A1_cv_r2", "A2_cv_r2_2024_26",
                                           "B1_forward_r2",
                                           "C1_calibration_slope")},
        "paired": pdl, "targeted": tg, "c2_worst": [seg, growth],
        "c1_gap": c1_gap, "b1_move": b1_move, "pass": bool(ok),
    }, indent=2), encoding="utf-8")
    pd.DataFrame({"player": df["player_name_norm"], "season": df["season"],
                  "affected": affected, "actual": y, "incumbent": ref.oof,
                  "kf_reprice": cand.oof}).to_csv(
        OUT / "kf_reprice_oof.csv", index=False)
    print(f"\nwrote {OUT / 'kf_reprice_eval.json'}")


if __name__ == "__main__":
    main()
