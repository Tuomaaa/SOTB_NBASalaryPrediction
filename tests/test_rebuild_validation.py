"""Regression tests for rebuild membership-aware feature validation."""

import pandas as pd
import pytest

from scripts.rebuild_training_data import validate


def _frame(extra_2026=False, raw_regression=False):
    rows = [
        {
            "player_name_norm": "old player",
            "season": 2025,
            "darko_dpm": 1.0,
            "lebron": 1.0,
            "laker": 1.0,
            "darko_dpm_z": 0.0,
            "lebron_z": 0.0,
            "laker_z": 0.0,
            "year_in_contract": 1,
        },
        {
            "player_name_norm": "current player",
            "season": 2026,
            "darko_dpm": 2.0 + int(raw_regression),
            "lebron": 2.0,
            "laker": 2.0,
            "darko_dpm_z": -1.0 if extra_2026 else 0.0,
            "lebron_z": -1.0 if extra_2026 else 0.0,
            "laker_z": -1.0 if extra_2026 else 0.0,
            "year_in_contract": 1,
        },
    ]
    if extra_2026:
        rows.append({
            "player_name_norm": "new signing",
            "season": 2026,
            "darko_dpm": 3.0,
            "lebron": 3.0,
            "laker": 3.0,
            "darko_dpm_z": 1.0,
            "lebron_z": 1.0,
            "laker_z": 1.0,
            "year_in_contract": 1,
        })
    return pd.DataFrame(rows)


def test_validator_allows_zscore_shift_only_in_expanded_season():
    validate(_frame(), _frame(extra_2026=True))


def test_validator_still_rejects_raw_impact_regression():
    with pytest.raises(SystemExit, match="darko_dpm"):
        validate(_frame(), _frame(extra_2026=True, raw_regression=True))
