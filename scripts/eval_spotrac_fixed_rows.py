"""Fixed-row HEAD versus Spotrac salary-migration comparison.

Both variants use the current evaluation code, supporting source tables,
GroupKFold split, and seeds. The only input that changes is
``training_data_v2.csv``:

* incumbent: the file stored at Git HEAD
* candidate: an explicitly supplied rebuilt Spotrac candidate

Each input first passes the normal evaluation filters. Scoring is then
restricted to the intersection of player-season keys, so row membership and
the R2 denominator cannot masquerade as model improvement or regression.

This harness is read-only with respect to formal model artifacts. It writes a
JSON report and a per-row CSV to the requested output directory.
"""

from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import src.model.evaluate_suite as suite
import src.model.train as train
from config import OUTPUTS_DIR


KEY = ["player_name_norm", "season"]


def _head_training() -> pd.DataFrame:
    raw = subprocess.check_output(
        ["git", "show", "HEAD:data/processed/training_data_v2.csv"]
    )
    return pd.read_csv(io.BytesIO(raw))


@contextmanager
def _training_source(frame: pd.DataFrame):
    """Make train.load_training_data read ``frame`` without moving files."""
    original = pd.read_csv

    def read_csv(path, *args, **kwargs):
        try:
            name = Path(path).name
        except TypeError:
            name = ""
        if name in {"training_data_v2.csv", "training_data.csv"}:
            return frame.copy(deep=True)
        return original(path, *args, **kwargs)

    with patch.object(pd, "read_csv", side_effect=read_csv):
        yield


def _evaluation_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    with _training_source(frame):
        return suite.load_evaluation_frame(
            allow_missing_computed=True, verbose=False
        )


def _fixed(frame: pd.DataFrame, keys: pd.MultiIndex) -> pd.DataFrame:
    indexed = frame.set_index(KEY, drop=False)
    missing = keys.difference(indexed.index)
    if len(missing):
        raise RuntimeError(f"fixed keys missing from frame: {list(missing)[:10]}")
    return indexed.loc[keys].reset_index(drop=True)


def _score(
    name: str,
    full_training: pd.DataFrame,
    fixed_eval: pd.DataFrame,
    base_features: list[str],
) -> tuple[suite.SuiteResult, pd.DataFrame]:
    print(f"\n{'=' * 78}\n{name}: {len(fixed_eval)} fixed rows\n{'=' * 78}")
    df, clf_features = suite.attach_clf_features(fixed_eval.copy())
    with _training_source(full_training):
        kf_ctx = suite.prepare_kf_context(
            df,
            base_features,
            prehistory=True,
            expand_anchors=True,
        )
        fitter = suite.make_kf_stage_arms_fitter(kf_ctx, clf_features)
        arms, _ = suite.run_suite_arms_kf(
            df,
            list(train.FEATURE_COLS),
            fitter,
            kf_ctx,
            clf_features,
            seeds=suite.DEFAULT_SEEDS,
            ladder={},
        )
    champion = arms[suite.ARM_CHAMPION]
    suite.print_report(df, champion)
    rows = df[KEY + [suite.TARGET, "salary_m", "signing_cat"]].copy()
    rows[f"{name}_oof"] = champion.oof
    rows[f"{name}_forward"] = champion.forward
    return champion, rows


def _metric_subset(result: suite.SuiteResult) -> dict:
    wanted = [
        "A1_cv_r2", "A1_cv_mae_m", "A1_cv_bias_m", "A1_n",
        "A2_cv_r2_2024_26", "A2_mae_m", "A2_bias_m", "A2_n",
        "B1_forward_r2", "B1_mae_m", "B1_bias_m", "B1_n",
        "B1_by_origin", "C1_calibration_slope",
        "C1_calibration_intercept", "C2_by_signing_mechanism",
        "D3_selection_r2", "D3_confirmation_r2",
    ]
    return {key: result.metrics.get(key) for key in wanted}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--candidate-seasons-from",
        type=int,
        help=(
            "Build an isolated candidate in memory: keep Git HEAD rows before "
            "this season and use candidate rows from this season onward."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUTS_DIR / "experiments" / "spotrac_fixed_rows",
    )
    args = parser.parse_args()

    head_training = _head_training()
    candidate_training = pd.read_csv(args.candidate)
    if args.candidate_seasons_from is not None:
        cutoff = args.candidate_seasons_from
        incumbent = head_training[head_training["season"] < cutoff]
        refreshed = candidate_training[candidate_training["season"] >= cutoff]
        candidate_training = pd.concat(
            [incumbent, refreshed], ignore_index=True
        )
        print(
            f"isolated candidate: HEAD seasons < {cutoff} ({len(incumbent)} rows), "
            f"candidate seasons >= {cutoff} ({len(refreshed)} rows)"
        )
    head_eval, head_features = _evaluation_frame(head_training)
    candidate_eval, candidate_features = _evaluation_frame(candidate_training)
    if head_features != candidate_features:
        raise SystemExit(
            f"feature mismatch: HEAD={head_features}, candidate={candidate_features}"
        )

    head_keys = pd.MultiIndex.from_frame(head_eval[KEY])
    candidate_keys = pd.MultiIndex.from_frame(candidate_eval[KEY])
    common = head_keys.intersection(candidate_keys).sort_values()
    print(
        f"HEAD evaluation rows={len(head_eval)}, candidate={len(candidate_eval)}, "
        f"fixed intersection={len(common)}, "
        f"HEAD-only={len(head_keys.difference(common))}, "
        f"candidate-only={len(candidate_keys.difference(common))}"
    )
    head_fixed = _fixed(head_eval, common)
    candidate_fixed = _fixed(candidate_eval, common)
    if not head_fixed[KEY].equals(candidate_fixed[KEY]):
        raise RuntimeError("fixed row order differs between variants")

    head_result, head_rows = _score(
        "head", head_training, head_fixed, head_features
    )
    candidate_result, candidate_rows = _score(
        "candidate", candidate_training, candidate_fixed, candidate_features
    )

    paired_all = suite.paired_delta(
        head_result.fold_r2, candidate_result.fold_r2
    )
    paired_selection = suite.paired_delta(
        head_result.fold_r2_sel, candidate_result.fold_r2_sel
    )
    target_delta_m = (
        candidate_fixed["salary_m"].to_numpy()
        - head_fixed["salary_m"].to_numpy()
    )
    changed = np.abs(target_delta_m) > 0.001
    report = {
        "description": (
            "HEAD versus isolated current-season Spotrac migration on fixed "
            "shared rows"
            if args.candidate_seasons_from is not None
            else "HEAD versus Spotrac migration on fixed shared rows"
        ),
        "candidate_seasons_from": args.candidate_seasons_from,
        "seeds": list(suite.DEFAULT_SEEDS),
        "n_splits": suite.N_SPLITS,
        "membership": {
            "head_filtered": len(head_eval),
            "candidate_filtered": len(candidate_eval),
            "fixed_intersection": len(common),
            "head_only": len(head_keys.difference(common)),
            "candidate_only": len(candidate_keys.difference(common)),
        },
        "targets": {
            "changed_rows": int(changed.sum()),
            "sum_abs_delta_m": float(np.abs(target_delta_m).sum()),
            "median_abs_changed_delta_m": (
                float(np.median(np.abs(target_delta_m[changed])))
                if changed.any() else 0.0
            ),
        },
        "head": _metric_subset(head_result),
        "candidate": _metric_subset(candidate_result),
        "paired_delta_candidate_minus_head": {
            "all_rows": paired_all,
            "selection_pool": paired_selection,
        },
    }

    rows = head_rows.merge(
        candidate_rows,
        on=KEY,
        validate="one_to_one",
        suffixes=("_head", "_candidate"),
    )
    rows["target_delta_m"] = (
        rows["salary_m_candidate"] - rows["salary_m_head"]
    )
    rows["head_oof_error_m"] = (
        rows["head_oof"] - rows["cap_pct_head"]
    ) * rows["salary_m_head"] / rows["cap_pct_head"]
    rows["candidate_oof_error_m"] = (
        rows["candidate_oof"] - rows["cap_pct_candidate"]
    ) * rows["salary_m_candidate"] / rows["cap_pct_candidate"]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "comparison.json"
    rows_path = args.output_dir / "rows.csv"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    rows.to_csv(rows_path, index=False)

    print("\nFIXED-ROW RESULT")
    for label, key in (
        ("A1", "A1_cv_r2"),
        ("A2", "A2_cv_r2_2024_26"),
        ("B1", "B1_forward_r2"),
    ):
        before = report["head"][key]
        after = report["candidate"][key]
        print(f"  {label}: {before:.4f} -> {after:.4f} ({after-before:+.4f})")
    d = report["paired_delta_candidate_minus_head"]["selection_pool"]
    print(
        f"  paired selection delta={d['delta']:+.5f} "
        f"SE={d['se']:.5f} t={d['t']:+.2f}"
    )
    print(f"wrote {report_path}")
    print(f"wrote {rows_path}")


if __name__ == "__main__":
    main()
