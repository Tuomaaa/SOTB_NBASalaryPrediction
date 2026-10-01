"""Score P(max) as a Stage-1 term against the Stage-2 push.

Pre-registered in `docs/briefs/2026-09-30-pmax-stage1.md`. Arms, all on the
champion pipeline (nested KF, Stage 3, signing offsets) with identical folds
and seeds:

    R   v6.2.0 champion: push, then clip
    N   no push: clip only
    F   latent = GBM(x) + waiver term + beta_max * z_max, no push, where
        z_max = P(max) * max(max_eligible_pct - kf_market_value, 0) and
        beta_max is fitted in each training slice (route_mixture.max_term)

The KF measurement model keeps the production composition (push included) in
every arm; only the scored model changes.

    python scripts/eval_pmax_stage1.py --count-only      # affected rows
    python scripts/eval_pmax_stage1.py --seeds 10 --full  # all layers
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
from src.model import route_mixture as rm
from src.model.evaluate_suite import (
    ARM_CHAMPION, ARM_CLIP, ARM_EXANTE, DEFAULT_SEEDS, abs_bias_growth,
    attach_clf_features, baseline_ladder, fold_splits, layer_a, layer_c,
    load_evaluation_frame, make_kf_stage_arms_fitter, oof_groupkfold_signing,
    paired_delta, prepare_kf_context, run_suite_arms_kf,
)
from src.model.train import FEATURE_COLS

OUT = OUTPUTS_DIR / "models"
# Explicit switches: the defaults moved to F in v6.3.0.
ARMS = {
    "R": {"push": True, "max_term_on": False},
    "N": {"push": False, "max_term_on": False},
    "F": {"push": False, "max_term_on": True},
}
NAMED = [("demar derozan", 2024), ("lamarcus aldridge", 2019),
         ("james harden", 2022), ("james harden", 2025),
         ("pascal siakam", 2020), ("devin booker", 2019),
         ("d'angelo russell", 2019)]
P_AFFECTED = 0.10
C2_BAR, C1_BAR, T_BAR = 0.30, 0.005, 2.0


def affected_rows(df: pd.DataFrame, clf_features: list[str],
                  seeds: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Non-waived selection rows with seed-mean OOF P(max) >= P_AFFECTED.

    Uses the suite's partitions (seed i on partition i) and the production
    classifier, so the P(max) is the one arm R composes with.
    """
    p = np.zeros(len(df))
    for si, seed in enumerate(seeds):
        for tr, va in fold_splits(df, si):
            clf = rm.train_route_classifier(df.iloc[tr], clf_features, seed)
            p[va] += rm.route_proba(clf, df.iloc[va], clf_features)[:, rm.MAX_IDX]
    p /= len(seeds)
    sel = ~df["is_confirmation"].astype(bool).values
    return sel & ~rm.waived_mask(df) & (p >= P_AFFECTED), p


def err_m(df: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
    """Signed error in $M."""
    return (pred - df["cap_pct"].values) * df["cap"].values / 1e6


def named_rows(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Signed error in $M on the brief's named rows."""
    e = err_m(df, pred)
    out = {}
    for player, season in NAMED:
        m = ((df["player_name_norm"] == player) & (df["season"] == season)).values
        if m.any():
            out[f"{player} {season}"] = round(float(e[np.flatnonzero(m)[0]]), 2)
    return out


def zones(df: pd.DataFrame, pred: np.ndarray) -> dict:
    """Bias and MAE in $M by zone on selection rows."""
    sel = ~df["is_confirmation"].astype(bool).values
    mx = df["is_max_contract"].astype(bool).values
    fl = df["is_at_floor"].astype(bool).values
    e = err_m(df, pred)
    out = {}
    for name, m in (("max", mx), ("floor", fl & ~mx), ("middle", ~mx & ~fl)):
        m = m & sel
        out[name] = {"n": int(m.sum()), "bias_m": float(e[m].mean()),
                     "mae_m": float(np.abs(e[m]).mean())}
    return out


def targeted_gate(df: pd.DataFrame, ref: np.ndarray, cand: np.ndarray,
                  affected: np.ndarray) -> dict:
    """Paired squared-error gain on affected rows, summed per player."""
    y = df["cap_pct"].values
    gain = (ref - y) ** 2 - (cand - y) ** 2
    per = pd.Series(gain[affected]).groupby(
        df["player_name_norm"].values[affected]).sum()
    se = per.std(ddof=1) / np.sqrt(len(per))
    return {"rows": int(affected.sum()), "players": int(len(per)),
            "mean_gain": float(per.mean()), "t": float(per.mean() / se),
            "share_improved": float((per > 0).mean())}


def main() -> None:
    """Score the arms and report the pre-registered gates."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--full", action="store_true",
                    help="all four layers (B1 included), not only layer A")
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--count-only", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])
    names = ["R"] + [a for a in args.arms if a != "R"]

    df, base_features = load_evaluation_frame(allow_missing_computed=True)
    df, clf_features = attach_clf_features(df)
    affected, p_mean = affected_rows(df, clf_features, seeds)
    print(f"frame {len(df)} rows; seeds {len(seeds)}; affected rows "
          f"(non-waived selection, seed-mean P(max) >= {P_AFFECTED}): "
          f"{int(affected.sum())} rows, "
          f"{df.loc[affected, 'player_name_norm'].nunique()} players; "
          f"P(max) >= 0.52: {int((affected & (p_mean >= 0.52)).sum())}",
          flush=True)
    if args.count_only:
        return

    kf_ctx = prepare_kf_context(df, base_features)
    ladder = baseline_ladder(df, base_features, seeds) if args.full else None
    features = list(FEATURE_COLS)
    results = {}
    for name in names:
        gp = ARMS[name]
        print(f"\nARM {name}: grabit_params {gp}", flush=True)
        rm.MAX_BETA_LOG.clear()
        fitter = make_kf_stage_arms_fitter(kf_ctx, clf_features,
                                           grabit_params=gp)
        if args.full:
            res, _ = run_suite_arms_kf(df, features, fitter, kf_ctx,
                                       clf_features, seeds=seeds,
                                       ladder=ladder, grabit_params=gp)
            r = res[ARM_CHAMPION]
            oof, sel_cells, metrics = r.oof, r.fold_r2_sel, dict(r.metrics)
            stages = {k: res[k].oof for k in (ARM_CLIP, ARM_EXANTE)}
        else:
            store, _ = oof_groupkfold_signing(df, features, fitter, seeds)
            oof, fold_r2, sel_cells = store[ARM_CHAMPION]
            metrics = {**layer_a(df, oof, fold_r2), **layer_c(df, oof)}
            stages = {k: store[k][0] for k in (ARM_CLIP, ARM_EXANTE)}
        betas = list(rm.MAX_BETA_LOG)
        if betas:
            print(f"  beta_max over {len(betas)} fits: mean {np.mean(betas):.3f} "
                  f"[{np.min(betas):.3f}, {np.max(betas):.3f}]", flush=True)
        results[name] = {"oof": oof, "sel": sel_cells, "m": metrics,
                         "stages": stages, "betas": betas}

    ref = results["R"]
    report = {}
    for name in names:
        r, m = results[name], results[name]["m"]
        e = {"A1": m["A1_cv_r2"], "A2": m["A2_cv_r2_2024_26"],
             "B1": m.get("B1_forward_r2"),
             "C1_slope": m["C1_calibration_slope"],
             "zones": zones(df, r["oof"]),
             "named_clip_exante_final_m": {
                 k: [named_rows(df, r["stages"][ARM_CLIP])[k],
                     named_rows(df, r["stages"][ARM_EXANTE])[k], v]
                 for k, v in named_rows(df, r["oof"]).items()}}
        if r["betas"]:
            b = np.array(r["betas"])
            lo, hi = rm.MAX_BETA_BOUNDS
            e["beta_max"] = {"mean": float(b.mean()), "min": float(b.min()),
                             "max": float(b.max()), "n": int(len(b)),
                             "at_lower": float(np.mean(b <= lo + 1e-6)),
                             "at_upper": float(np.mean(b >= hi - 1e-6))}
        if name != "R":
            pdl = paired_delta(ref["sel"], r["sel"])
            a, c = ref["m"]["C2_by_signing_mechanism"], m["C2_by_signing_mechanism"]
            growth = {k: abs_bias_growth(c[k]["bias_m"], a[k]["bias_m"])
                      for k in set(a) & set(c)}
            worst = max(growth, key=growth.get)
            tg = targeted_gate(df, ref["oof"], r["oof"], affected)
            b1_ok = (e["B1"] is None or ref["m"].get("B1_forward_r2") is None
                     or e["B1"] - ref["m"]["B1_forward_r2"] >= 0)
            c1_gap = (abs(m["C1_calibration_slope"] - 1)
                      - abs(ref["m"]["C1_calibration_slope"] - 1))
            e.update({"dSel": pdl["delta"], "se": pdl["se"], "t": pdl["t"],
                      "per_fold": pdl["per_fold"], "targeted": tg,
                      "C2_worst": worst, "C2_growth_m": float(growth[worst]),
                      "C1_slope_gap": float(c1_gap), "B1_ok": bool(b1_ok)})
            e["gates"] = {
                "targeted_t": tg["t"] > T_BAR, "dSel_pos": pdl["delta"] > 0,
                "B1": bool(b1_ok),
                "C1_C2": c1_gap <= C1_BAR and growth[worst] <= C2_BAR}
            e["pass"] = all(e["gates"].values())
        report[name] = e

    for name in names:
        e = report[name]
        b1 = "" if e["B1"] is None else f"  B1 {e['B1']:.4f}"
        print(f"\n{name}: A1 {e['A1']:.4f}  A2 {e['A2']:.4f}{b1}  "
              f"slope {e['C1_slope']:.3f}")
        if name != "R":
            tg = e["targeted"]
            print(f"  dSel {e['dSel']:+.5f} se {e['se']:.5f} t {e['t']:+.2f}  "
                  f"per fold {e['per_fold']}")
            print(f"  targeted: {tg['rows']} rows / {tg['players']} players, "
                  f"t {tg['t']:+.2f}, improved {tg['share_improved']:.0%}")
            print(f"  C2 worst {e['C2_worst']} {e['C2_growth_m']:+.3f}M  "
                  f"C1 gap {e['C1_slope_gap']:+.4f}  gates {e['gates']} -> "
                  f"{'PASS' if e['pass'] else 'FAIL'}")
        if "beta_max" in e:
            print(f"  beta_max {e['beta_max']}")
        print(f"  zones {json.dumps(e['zones'])}")
        print(f"  named clip / exante / final ($M): "
              f"{e['named_clip_exante_final_m']}")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("full" if args.full else "screen")
    (OUT / f"pmax_stage1_{tag}.json").write_text(
        json.dumps({"seeds": list(seeds), "affected_rows": int(affected.sum()),
                    "arms": report}, indent=2), encoding="utf-8")
    pd.DataFrame({"player": df["player_name_norm"], "season": df["season"],
                  "actual": df["cap_pct"], "affected": affected,
                  "p_max_mean": p_mean,
                  **{n: results[n]["oof"] for n in names}}).to_csv(
        OUT / f"pmax_stage1_{tag}_oof.csv", index=False)
    print(f"\nwrote {OUT / f'pmax_stage1_{tag}.json'}")


if __name__ == "__main__":
    main()
