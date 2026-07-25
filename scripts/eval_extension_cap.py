"""Part 1 evidence for the extension route: classification, hard gate, headroom.

Prints, in order:

  1. the extension classification and its counts, with the rs-set instrument
     checked against the "rookie" keyword it replaces;
  2. the knife-edge table — rows landing on the raise multiple to the dollar,
     which is what validates the rule against the data;
  3. the HARD GATE: rows paid above their own computed cap, with the diagnosis
     for each survivor;
  4. the honest headroom: how many rows the champion prices above the corrected
     cap and the oracle dR2 from snapping exactly those rows to it.

Run:  OMP_NUM_THREADS=6 python scripts/eval_extension_cap.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import CAP_BY_SEASON, OUTPUTS_DIR
from src.model.evaluate_suite import (
    load_evaluation_frame, make_grabit_fitter, oof_groupkfold, paired_delta,
    _dollars, DEFAULT_SEEDS, TARGET,
)
from src.model.extension_cap import attach_extension_cap, over_cap_rows, CAP_TOL


def main():
    pd.set_option("display.width", 260)
    df, features = load_evaluation_frame()
    df = attach_extension_cap(df)
    cap = df["season"].map(CAP_BY_SEASON).values
    y = df[TARGET].values
    report = {"n_rows": len(df)}

    # ---- 1. classification ------------------------------------------------
    ext = df[df["is_extension"]]
    print("\n" + "=" * 74)
    print("  PART 1a — extension classification")
    print("=" * 74)
    print(f"  frame {len(df)} rows | first-year extension rows {len(ext)}")
    print(ext.groupby(["ext_kind"]).size().to_string())
    print("\n  by paying season:")
    print(pd.crosstab(ext["season"], ext["ext_kind"]).to_string())
    print(f"\n  designated-veteran (raise cap does not apply): "
          f"{int(ext['ext_is_dvp'].sum())}")
    print("    " + ", ".join(f"{r.player_name_norm} {int(r.season)}"
                             for r in ext[ext.ext_is_dvp].itertuples()))
    report["n_extension"] = int(len(ext))
    report["by_kind"] = ext.groupby("ext_kind").size().to_dict()
    report["n_dvp"] = int(ext["ext_is_dvp"].sum())

    # instrument check: the rs-set classification vs the keyword it replaces
    from scripts.parse_signing_dates import contract_spans, covering_contract
    spans = contract_spans()
    txt = []
    for i in np.flatnonzero(df["is_extension"].values):
        cov = covering_contract(spans, df["player_name_norm"].iat[i],
                                int(df["season"].iat[i]),
                                salary=float(y[i]) * cap[i])
        txt.append(str(cov["tx_text"]))
    kw = pd.Series(txt).str.contains("rookie", case=False).values
    rs = (ext["ext_kind"] == "rookie_scale").values
    print(f"\n  instrument check vs the 'rookie' keyword: agree {int((kw == rs).sum())}"
          f"/{len(ext)}, disagree {int((kw != rs).sum())}")
    for j in np.flatnonzero(kw != rs):
        r = ext.iloc[j]
        print(f"    {r.player_name_norm:22s} {int(r.season)}  rs_set="
              f"{rs[j]!s:5s} keyword={kw[j]!s:5s}  {txt[j][:95]}")
    report["keyword_disagreements"] = int((kw != rs).sum())

    # ---- 2. knife edge ----------------------------------------------------
    print("\n" + "=" * 74)
    print("  PART 1b — the knife edge (dollar ratio vs the legal multiple)")
    print("=" * 74)
    v = ext[ext["ext_kind"] == "veteran"].copy()
    v["pay_m"] = v[TARGET] * v["season"].map(CAP_BY_SEASON) / 1e6
    v["prior_m"] = v["ext_prior_usd"] / 1e6
    v["ratio"] = v["pay_m"] / v["prior_m"]
    v["mult"] = np.where(v["ext_sign_season"] >= 2023, 1.40, 1.20)
    hit = v[(v["ratio"] - v["mult"]).abs() < 0.005]
    print(f"  veteran extensions {len(v)}; paid at EXACTLY the multiple: {len(hit)}")
    print(hit[["player_name_norm", "season", "ext_sign_season", "pay_m", "prior_m",
               "ratio", "mult"]].to_string(index=False))
    eas = v[(v["pay_m"] * 1e6 - v["ext_cap_pct"] * v["season"].map(CAP_BY_SEASON)
             ).abs() < 5000]
    print(f"\n  paid at their computed cap to within $5k (either route): {len(eas)}")
    print(eas[["player_name_norm", "season", "ext_sign_season", "pay_m",
               "prior_m", "ratio"]].to_string(index=False))
    report["n_exactly_at_multiple"] = int(len(hit))
    report["n_at_cap_within_5k"] = int(len(eas))

    # ---- 3. HARD GATE -----------------------------------------------------
    print("\n" + "=" * 74)
    print("  PART 1c — HARD GATE: rows paid above their own computed cap")
    print("=" * 74)
    bad = over_cap_rows(df)
    print(f"  over-cap rows: {len(bad)} of {len(ext)} extension rows "
          f"(tolerance {CAP_TOL:g} cap_pct)")
    if len(bad):
        print(bad[["player_name_norm", "season", "pay_m", "cap_m", "over_m",
                   "ext_kind", "ext_sign_season", "ext_is_dvp"]].to_string(index=False))
        raw = pd.read_csv(Path(__file__).resolve().parent.parent /
                          "data/processed/training_data_v2.csv",
                          usecols=["player_name_norm", "season", "salary"])
        print("\n  diagnosis — the prior salary each row's pay implies:")
        for r in bad.itertuples():
            mult = 1.40 if r.ext_sign_season >= 2023 else 1.20
            h = raw[raw.player_name_norm == r.player_name_norm].sort_values("season")
            line = " ".join(f"{int(x.season)}:{x.salary/1e6:.2f}" for x in h.itertuples())
            print(f"    {r.player_name_norm:20s} {int(r.season)}  implied prior "
                  f"${r.pay_m/mult:.2f}M vs table ${r.ext_prior_usd/1e6:.2f}M "
                  f"(gap ${r.pay_m/mult - r.ext_prior_usd/1e6:+.2f}M)   [{line}]")
    report["over_cap_n"] = int(len(bad))
    report["over_cap_rows"] = bad.to_dict("records") if len(bad) else []

    # ---- 4. honest headroom ----------------------------------------------
    print("\n" + "=" * 74)
    print("  PART 1d — honest headroom (oracle: snap over-cap predictions down)")
    print("=" * 74)
    champ_oof, frs, frs_sel = oof_groupkfold(df, features, make_grabit_fitter(),
                                             DEFAULT_SEEDS)
    ec = df["ext_cap_pct"].values
    has = ~np.isnan(ec)
    above = has & (champ_oof > ec + CAP_TOL)
    snapped = champ_oof.copy()
    snapped[above] = ec[above]

    a1_c, a1_s = r2_score(y, champ_oof), r2_score(y, snapped)
    mae_c, bias_c = _dollars(df, champ_oof)
    mae_s, bias_s = _dollars(df, snapped)
    print(f"  champion A1 {a1_c:.4f}  ->  snapped {a1_s:.4f}   "
          f"(oracle dR2 {a1_s - a1_c:+.5f})")
    print(f"  champion MAE ${mae_c:.3f}M -> ${mae_s:.3f}M;  "
          f"bias ${bias_c:+.3f}M -> ${bias_s:+.3f}M")
    print(f"  rows priced above their corrected cap: {int(above.sum())} "
          f"of {int(has.sum())} extension rows")
    if above.sum():
        d = pd.DataFrame({
            "player": df["player_name_norm"].values[above],
            "season": df["season"].values[above],
            "kind": df["ext_kind"].values[above],
            "pay_m": y[above] * cap[above] / 1e6,
            "pred_m": champ_oof[above] * cap[above] / 1e6,
            "cap_m": ec[above] * cap[above] / 1e6,
        })
        d["cut_m"] = d["pred_m"] - d["cap_m"]
        d["err_before_m"] = d["pred_m"] - d["pay_m"]
        d["err_after_m"] = d["cap_m"] - d["pay_m"]
        d["gain_m"] = d["err_before_m"].abs() - d["err_after_m"].abs()
        print(d.sort_values("cut_m", ascending=False).to_string(index=False))
        print(f"\n  total prediction cut ${d['cut_m'].sum():.1f}M over "
              f"{len(d)} rows; MAE gain on those rows "
              f"${d['gain_m'].mean():+.2f}M/row")
    report["headroom"] = {
        "A1_champion": float(a1_c), "A1_snapped": float(a1_s),
        "oracle_dR2": float(a1_s - a1_c), "n_above": int(above.sum()),
        "n_extension_with_cap": int(has.sum()),
        "mae_champion": mae_c, "mae_snapped": mae_s,
    }

    # veteran-only variant: the rookie-scale rows keep the tier ceiling, so
    # they cannot move; report the veteran slice on its own.
    vet_mask = (df["ext_kind"] == "veteran").values & has
    vab = vet_mask & (champ_oof > ec + CAP_TOL)
    print(f"\n  veteran-extension slice only: {int(vab.sum())} rows above cap "
          f"of {int(vet_mask.sum())}")
    if vab.sum():
        snap_v = champ_oof.copy()
        snap_v[vab] = ec[vab]
        print(f"    oracle dR2 (veteran rows only) "
              f"{r2_score(y, snap_v) - a1_c:+.5f}")
        report["headroom"]["oracle_dR2_veteran_only"] = float(
            r2_score(y, snap_v) - a1_c)
        report["headroom"]["n_above_veteran"] = int(vab.sum())

    out = OUTPUTS_DIR / "models"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "extension_cap_eval.json", "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    dump = df[["player_name_norm", "season", TARGET, "signing_cat",
               "max_eligible_pct", "tier_ceiling_pct", "is_extension", "ext_kind",
               "ext_sign_season", "ext_is_dvp", "ext_prior_usd", "ext_cap_pct",
               "is_confirmation"]].copy()
    dump["oof_champion"] = champ_oof
    dump.to_csv(out / "extension_cap_oof.csv", index=False)
    print(f"\nSaved {out / 'extension_cap_eval.json'}")
    print(f"Saved {out / 'extension_cap_oof.csv'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
