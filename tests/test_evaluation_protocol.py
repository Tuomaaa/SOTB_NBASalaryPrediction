"""Repeated grouped CV: stable fold map and the corrected paired statistic."""

import numpy as np
import pandas as pd

from src.model.evaluate_suite import N_SPLITS, fold_ids, fold_splits, paired_delta


def test_fold_of_a_player_ignores_other_rows():
    players = [f"player {i}" for i in range(200)]
    full = fold_ids(players, 0)
    fewer = fold_ids(players[:150], 0)
    assert (full[:150] == fewer).all()


def test_partitions_differ_and_cover_every_row():
    df = pd.DataFrame({"player_name_norm": [f"p{i}" for i in range(300)]})
    a, b = fold_ids(df.player_name_norm, 0), fold_ids(df.player_name_norm, 1)
    assert (a != b).mean() > 0.5
    splits = fold_splits(df, 0)
    assert len(splits) == N_SPLITS
    test_rows = np.concatenate([va for _, va in splits])
    assert sorted(test_rows) == list(range(len(df)))


def test_paired_delta_uses_repeated_cv_correction():
    rng = np.random.default_rng(0)
    a = rng.normal(0.8, 0.03, size=(N_SPLITS, 10))
    b = a + 0.01 + rng.normal(0, 0.005, size=a.shape)
    out = paired_delta(a, b)
    d = (b - a).ravel()
    naive_se = d.std(ddof=1) / np.sqrt(d.size)
    assert abs(out["delta"] - d.mean()) < 1e-12
    assert out["se"] > 2 * naive_se
