"""Remeasure the waiver features and test kf_market_value_x_waived.

Arms, all scored on the champion pipeline (nested KF, Stage 2/3, signing
offsets) with identical folds and seeds. `incumbent` is the current champion
(v6.2.0, term_flags_owed); `v606`, `v610` and every historical arm carry their
feature list and switches explicitly, so their definitions do not move:

    incumbent         FEATURE_COLS, waiver term on money-owed waivers,
                      P(max) = 0 on waived rows
    v606              v6.0.6 champion (is_waived and mpg_x_waived features)
    v610              v6.1.0 champion (waiver_term_noflag)
    no_is_waived      without is_waived
    no_mpg_x_waived   without mpg_x_waived
    kf_x_waived       + is_waived * kf_market_value
    kf_x_known        + is_waived_known * kf_market_value (coverage control)
    term_flags        v6.1.0 champion with is_waived and mpg_x_waived kept
    term_flags_owed   term_flags, term applied to money-owed waivers only
    waiver_term       partially linear Stage 1: latent = GBM(x) + beta * z,
                      z = is_waived * kf_market_value,
                      beta fitted in each training slice (see
                      route_mixture.waiver_beta; not adopted, v6.0.3)
    owed_branch       money-owed waivers leave the Grabit fit and are priced
                      as gamma * latent (Tobit, left-censored at the floor);
                      P(max) = 0 on waived rows (pre-registered in QUEUE)

The interactions are built after nested KF inference inside every fold
(`augment`), so no fold sees a kf value informed by its own players. The KF
measurement model (MEASUREMENT_FEATURES) and the route classifier keep their
production feature lists in every arm; only the regression features change.

    python scripts/eval_waiver_challengers.py --seeds 3            # layer A screen
    python scripts/eval_waiver_challengers.py --seeds 10 --full \\
        --arms incumbent kf_x_waived kf_x_known                    # all layers
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
from src.model import route_mixture as rm
from src.model.evaluate_suite import (
    ARM_CHAMPION, ARM_CLIP, ARM_EXANTE, DEFAULT_SEEDS, abs_bias_growth, attach_clf_features,
    baseline_ladder, layer_a, layer_c, load_evaluation_frame,
    make_kf_stage_arms_fitter, oof_groupkfold_signing, paired_delta,
    prepare_kf_context, run_suite_arms_kf,
)
from src.model.train import FEATURE_COLS

OUT = OUTPUTS_DIR / "models"
NAMED = [("kemba walker", 2021), ("hassan whiteside", 2020),
         ("damian lillard", 2025), ("andre drummond", 2021),
         ("bradley beal", 2025)]
C2_BAR = 0.30
DSEL_BAR, T_BAR = 0.002, 2.0


def _times_kf(source: str, name: str):
    """Frame transform adding `name` = source * kf_market_value."""
    def augment(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out[name] = (pd.to_numeric(out[source], errors="coerce").fillna(0.0)
                     * out["kf_market_value"])
        return out
    return augment


# v6.0.6 regression list, in its original order (column order changes which
# columns colsample_bytree draws).
V606_FEATURES = [
    "darko_dpm_z", "lebron_z", "laker_z", "age", "age_squared", "mpg",
    "availability_3yr", "is_waived", "usage_pct", "height_inches", "cba_era",
    "ast_pct", "award_score_cum", "draft_pick", "kf_market_value",
    "mpg_x_waived", "playoff_mpg_diff", "kalman_filtered_stats",
    "darko_od_diff_z", "lebron_od_diff_z", "laker_od_diff_z",
]
V606 = {"waiver_term": False, "exclude_waived_max": False}


def arms() -> dict:
    """Arm name -> (regression features, augment or None, grabit params)."""
    base = list(V606_FEATURES)
    return {
        "incumbent": (list(FEATURE_COLS), None, None),
        "v606": (base, None, V606),
        "v610": ([f for f in base if f not in ("is_waived", "mpg_x_waived")],
                 None, {"waiver_term": True, "exclude_waived_max": True}),
        "no_is_waived": ([f for f in base if f != "is_waived"], None, V606),
        "no_mpg_x_waived": ([f for f in base if f != "mpg_x_waived"], None,
                            V606),
        "kf_x_waived": (base + ["kf_market_value_x_waived"],
                        _times_kf("is_waived", "kf_market_value_x_waived"),
                        V606),
        "kf_x_known": (base + ["kf_market_value_x_known"],
                       _times_kf("is_waived_known", "kf_market_value_x_known"),
                       V606),
        "waiver_term": (base, None, {**V606, "waiver_term": True}),
        # The v6.1.0 champion with the two waiver features kept: isolates
        # the effect of dropping them.
        "term_flags": (base, None, {"waiver_term": True,
                                    "exclude_waived_max": True}),
        # term_flags with the term applied to money-owed waivers only; beta
        # is still fitted on every waived row.
        "term_flags_owed": (base, None, {"waiver_term": "owed",
                                         "exclude_waived_max": True}),
        "owed_branch": (base, None, {**V606, "owed_branch": True,
                                     "exclude_waived_max": True}),
        "owed_censor": (base, None, {**V606, "owed_branch": "censor",
                                     "exclude_waived_max": True}),
        "waived_branch": (base, None, {**V606, "waived_branch": True,
                                       "exclude_waived_max": True}),
        "waiver_term_noflag": (
            [f for f in base if f not in ("is_waived", "mpg_x_waived")],
            None, {"waiver_term": True, "exclude_waived_max": True}),
    }


def c2_worst(df: pd.DataFrame, ref: dict, cand: dict) -> tuple[str, float]:
    """Largest |bias| growth over signing mechanisms, candidate vs reference."""
    a, b = ref["C2_by_signing_mechanism"], cand["C2_by_signing_mechanism"]
    growth = {k: abs_bias_growth(b[k]["bias_m"], a[k]["bias_m"])
              for k in set(a) & set(b)}
    worst = max(growth, key=growth.get)
    return worst, float(growth[worst])


def named_rows(df: pd.DataFrame, oof: np.ndarray) -> dict:
    """Signed error in $M on the rows the queue item names."""
    out = {}
    for player, season in NAMED:
        m = ((df["player_name_norm"] == player)
             & (df["season"] == season)).values
        if m.any():
            i = int(np.flatnonzero(m)[0])
            out[f"{player} {season}"] = round(
                float((oof[i] - df["cap_pct"].iat[i]) * df["cap"].iat[i] / 1e6), 2)
    return out


def targeted_gate(df: pd.DataFrame, ref: np.ndarray, cand: np.ndarray,
                  affected: np.ndarray) -> dict:
    """Targeted gate: paired squared-error gain on affected selection rows.

    Gains are summed per player (rows of one player are not independent) and
    tested across players. See `docs/worker-brief.md`.
    """
    y = df["cap_pct"].values
    m = affected & ~df["is_confirmation"].astype(bool).values
    gain = (ref - y) ** 2 - (cand - y) ** 2
    per = pd.Series(gain[m]).groupby(df["player_name_norm"].values[m]).sum()
    se = per.std(ddof=1) / np.sqrt(len(per))
    return {"rows": int(m.sum()), "players": int(len(per)),
            "mean_gain": float(per.mean()), "t": float(per.mean() / se),
            "share_improved": float((per > 0).mean())}


def main() -> None:
    """Score the requested arms and report the paired gates."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--full", action="store_true",
                    help="all four layers (B1 included), not only layer A")
    ap.add_argument("--arms", nargs="+", default=list(arms()))
    ap.add_argument("--tag", default=None,
                    help="artifact name suffix (default: full or screen)")
    args = ap.parse_args()
    seeds = tuple(DEFAULT_SEEDS[:args.seeds])
    spec = arms()
    names = ["incumbent"] + [a for a in args.arms if a != "incumbent"]

    df, base_features = load_evaluation_frame(allow_missing_computed=True)
    df, clf_features = attach_clf_features(df)
    kf_ctx = prepare_kf_context(df, base_features)
    ladder = baseline_ladder(df, base_features, seeds) if args.full else None
    pos = int(df["is_waived"].sum())
    known = int(df["is_waived_known"].sum())
    print(f"frame {len(df)} rows; is_waived positives {pos}; known {known}; "
          f"seeds {len(seeds)}; {'full suite' if args.full else 'layer A'}")

    results = {}
    for name in names:
        features, augment, gp = spec[name]
        print(f"\nARM {name}: {len(features)} regression features", flush=True)
        rm.WAIVER_BETA_LOG.clear()
        rm.OWED_GAMMA_LOG.clear()
        rm.EXCLUDED_PUSH_LOG.clear()
        rm.WAIVED_GAMMA_LOG.clear()
        fitter = make_kf_stage_arms_fitter(kf_ctx, clf_features,
                                           grabit_params=gp, augment=augment)
        if args.full:
            res, _ = run_suite_arms_kf(df, features, fitter, kf_ctx,
                                       clf_features, seeds=seeds,
                                       ladder=ladder, grabit_params=gp,
                                       augment=augment)
            r = res[ARM_CHAMPION]
            oof, sel_cells, metrics = r.oof, r.fold_r2_sel, dict(r.metrics)
            stages = {k: res[k].oof for k in (ARM_CLIP, ARM_EXANTE)}
        else:
            store, _ = oof_groupkfold_signing(df, features, fitter, seeds)
            oof, fold_r2, sel_cells = store[ARM_CHAMPION]
            metrics = {**layer_a(df, oof, fold_r2), **layer_c(df, oof)}
            stages = {k: store[k][0] for k in (ARM_CLIP, ARM_EXANTE)}
        betas = list(rm.WAIVER_BETA_LOG) or list(rm.OWED_GAMMA_LOG)
        if rm.WAIVED_GAMMA_LOG:
            g = np.array(rm.WAIVED_GAMMA_LOG)
            print(f"  gamma owed mean {g[:, 0].mean():.3f} "
                  f"[{g[:, 0].min():.3f}, {g[:, 0].max():.3f}]; plain mean "
                  f"{g[:, 1].mean():.3f} [{g[:, 1].min():.3f}, "
                  f"{g[:, 1].max():.3f}] over {len(g)} fits", flush=True)
        if rm.EXCLUDED_PUSH_LOG:
            print(f"  exclusion removed the push on "
                  f"{sum(rm.EXCLUDED_PUSH_LOG)} waived test rows over "
                  f"{len(rm.EXCLUDED_PUSH_LOG)} fits", flush=True)
        if betas:
            print(f"  beta/gamma over {len(betas)} fits: mean {np.mean(betas):+.3f} "
                  f"min {np.min(betas):+.3f} max {np.max(betas):+.3f}",
                  flush=True)
        results[name] = {"oof": oof, "sel": sel_cells, "m": metrics,
                         "stages": stages, "betas": betas}

    ref = results["incumbent"]
    conf = df["is_confirmation"].astype(bool)
    rows, report = [], {}
    for name in names:
        r = results[name]
        m = r["m"]
        entry = {
            "A1": m["A1_cv_r2"], "A2": m["A2_cv_r2_2024_26"],
            "B1": m.get("B1_forward_r2"),
            "C1_slope": m["C1_calibration_slope"],
            "canary_r2": float(r2_score(
                df.loc[conf, "cap_pct"], r["oof"][conf.values])),
            "named_err_m": named_rows(df, r["oof"]),
            "named_clip_push_final_m": {
                k: [named_rows(df, r["stages"][ARM_CLIP])[k],
                    named_rows(df, r["stages"][ARM_EXANTE])[k], v]
                for k, v in named_rows(df, r["oof"]).items()},
            "beta": ({"mean": float(np.mean(r["betas"])),
                      "min": float(np.min(r["betas"])),
                      "max": float(np.max(r["betas"])), "n": len(r["betas"])}
                     if r["betas"] else None),
        }
        if name != "incumbent":
            pdl = paired_delta(ref["sel"], r["sel"])
            seg, growth = c2_worst(df, ref["m"], m)
            entry.update({
                "dSel": pdl["delta"], "se": pdl["se"], "t": pdl["t"],
                "per_fold": pdl["per_fold"],
                "C2_worst": seg, "C2_growth_m": growth,
                "C1_slope_gap": abs(m["C1_calibration_slope"] - 1)
                                - abs(ref["m"]["C1_calibration_slope"] - 1),
            })
        report[name] = entry
        b1 = "" if entry["B1"] is None else f"  B1 {entry['B1']:.4f}"
        line = (f"{name:16s} A1 {entry['A1']:.4f}  A2 {entry['A2']:.4f}{b1}"
                f"  slope {entry['C1_slope']:.3f}")
        if name != "incumbent":
            line += (f"\n{'':16s} dSel {entry['dSel']:+.5f}  se {entry['se']:.5f}"
                     f"  t {entry['t']:+.2f}  C2 worst {entry['C2_worst']} "
                     f"{entry['C2_growth_m']:+.3f}M")
        if name != "incumbent":
            sel = ~conf.values
            y = df["cap_pct"].values
            d_sel = (r2_score(y[sel], r["oof"][sel])
                     - r2_score(y[sel], ref["oof"][sel]))
            d_can = entry["canary_r2"] - report["incumbent"]["canary_r2"]
            if abs(d_can - d_sel) > 0.005:
                line += (f"\n{'':16s} CANARY DIVERGENCE: selection {d_sel:+.4f}, "
                         f"confirmation {d_can:+.4f}")
        line += (f"\n{'':16s} named errors ($M) clip / push / final: "
                 f"{entry['named_clip_push_final_m']}")
        rows.append(line)

    print("\n" + "\n".join(rows))
    waived = (pd.to_numeric(df["is_waived"], errors="coerce")
              .fillna(0.0).values == 1.0)
    for name in names[1:]:
        e = report[name]
        ok = e["dSel"] >= DSEL_BAR and e["t"] > T_BAR and e["C2_growth_m"] <= C2_BAR
        print(f"  {name:16s} selection+C2 gate: {'PASS' if ok else 'FAIL'}")
        tg = targeted_gate(df, ref["oof"], results[name]["oof"], waived)
        e["targeted_waived"] = tg
        b1_ok = (e["B1"] is None or ref["m"].get("B1_forward_r2") is None
                 or e["B1"] - ref["m"]["B1_forward_r2"] >= 0)
        ok_t = (tg["t"] > T_BAR and e["dSel"] > 0 and b1_ok
                and e["C2_growth_m"] <= C2_BAR and e["C1_slope_gap"] <= 0.005)
        print(f"  {name:16s} targeted (waived selection rows): "
              f"{tg['rows']} rows / {tg['players']} players, t {tg['t']:+.2f}, "
              f"improved {tg['share_improved']:.0%} -> "
              f"{'PASS' if ok_t else 'FAIL'}")
        # Gate without the targeted t, set by the user for waiver_term_noflag.
        ok_u = (e["dSel"] > 0 and b1_ok and e["C2_growth_m"] <= C2_BAR
                and e["C1_slope_gap"] <= 0.005)
        print(f"  {name:16s} dSel>0 + B1 + C2 + C1 gate (targeted t "
              f"reported only): {'PASS' if ok_u else 'FAIL'}  "
              f"[dSel {e['dSel']:+.5f}, B1 ok {b1_ok}, C2 "
              f"{e['C2_growth_m']:+.3f}M, C1 gap {e['C1_slope_gap']:+.4f}]")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ("full" if args.full else "screen")
    (OUT / f"waiver_challengers_{tag}.json").write_text(
        json.dumps({"seeds": list(seeds), "arms": report}, indent=2),
        encoding="utf-8")
    pd.DataFrame({"player": df["player_name_norm"], "season": df["season"],
                  "actual": df["cap_pct"],
                  **{n: results[n]["oof"] for n in names}}).to_csv(
        OUT / f"waiver_challengers_{tag}_oof.csv", index=False)
    print(f"\nwrote {OUT / f'waiver_challengers_{tag}.json'}")


if __name__ == "__main__":
    main()
