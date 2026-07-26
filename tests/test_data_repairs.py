"""Focused regression tests for impact identity repairs."""

import unittest

import numpy as np
import pandas as pd

from config import CAP_BY_SEASON
from src.features.build_dataset import _resolve_merged_age
from src.features.availability import compute_availability
from src.features.impact_identity import (
    coalesce_impact_rows,
    fill_age_from_player_history,
    fill_impact_from_bbref,
)


class ImpactIdentityTests(unittest.TestCase):
    def test_coalesces_name_variants_by_nba_id(self):
        frame = pd.DataFrame([
            {
                "player_name": "Name Jr.", "player_name_norm": "name jr.",
                "nba_id": 123, "season": 2023, "darko_dpm": 1.2,
                "age": np.nan, "games": np.nan,
            },
            {
                "player_name": "Name", "player_name_norm": "name",
                "nba_id": "123", "season": 2023, "darko_dpm": np.nan,
                "age": 25, "games": 70,
            },
        ])

        result = coalesce_impact_rows(frame, {("name", 2023)})

        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "player_name_norm"], "name")
        self.assertEqual(result.loc[0, "darko_dpm"], 1.2)
        self.assertEqual(result.loc[0, "age"], 25)

    def test_fills_age_from_player_history_not_league_median(self):
        frame = pd.DataFrame({
            "player_name_norm": ["veteran", "veteran", "other"],
            "season": [2022, 2023, 2023],
            "age": [34.0, np.nan, 23.0],
        })

        result = fill_age_from_player_history(frame)

        self.assertEqual(result.loc[1, "age"], 35.0)

    def test_static_contract_page_age_uses_latest_joined_season_as_anchor(self):
        frame = pd.DataFrame({
            "player_name_norm": ["veteran"] * 3,
            "season": [2024, 2025, 2026],
            "age_imp": [np.nan, np.nan, np.nan],
            "age_sal": [np.nan, 34.0, 34.0],
        })

        frame["age"] = _resolve_merged_age(frame)
        result = fill_age_from_player_history(frame)

        self.assertEqual(result["age"].tolist(), [32.0, 33.0, 34.0])

    def test_bbref_fallback_fills_only_missing_workload(self):
        impact = pd.DataFrame({
            "player_name_norm": ["player"], "season": [2023],
            "age": [25.0], "games": [np.nan], "minutes": [np.nan],
            "usage_pct": [18.0], "ast_pct": [np.nan],
        })
        bbref = pd.DataFrame({
            "player_name_norm": ["player"], "season": [2023],
            "age": [26.0], "games": [70.0], "minutes": [1400.0],
            "usg_pct": [22.0], "ast_pct": [14.0],
        })

        result = fill_impact_from_bbref(impact, bbref)

        self.assertEqual(result.loc[0, "age"], 25.0)
        self.assertEqual(result.loc[0, "games"], 70.0)
        self.assertEqual(result.loc[0, "minutes"], 1400.0)
        self.assertEqual(result.loc[0, "usage_pct"], 18.0)
        self.assertEqual(result.loc[0, "ast_pct"], 14.0)

    def test_repaired_floor_crash_rows_have_observed_identity_and_workload(self):
        frame = pd.read_csv("data/processed/training_data_v2.csv").set_index(
            ["player_name_norm", "season"]
        )
        expected = {
            ("montrezl harrell", 2023): (29.0, 11.9),
            ("demarcus cousins", 2020): (29.0, 0.0),
            ("bol bol", 2023): (23.0, 21.5),
            ("malik beasley", 2023): (26.0, 25.8),
            ("patrick beverley", 2023): (34.0, 27.1),
        }

        for key, (age, mpg) in expected.items():
            self.assertEqual(float(frame.loc[key, "age"]), age)
            self.assertAlmostEqual(float(frame.loc[key, "mpg"]), mpg, places=1)
            self.assertFalse(pd.isna(frame.loc[key, "availability_3yr"]))

        beverley_prior = (
            float(frame.loc[("patrick beverley", 2023), "prev_cap_pct"])
            * CAP_BY_SEASON[2022]
        )
        self.assertAlmostEqual(beverley_prior, 13_000_000, places=0)


class AvailabilityTests(unittest.TestCase):
    def test_skips_unknown_seasons_but_keeps_observed_zero(self):
        frame = pd.DataFrame({
            "player_name_norm": ["player"] * 4,
            "season": [2020, 2021, 2022, 2023],
            "games": [np.nan, 36.0, 0.0, 82.0],
        })

        result = compute_availability(frame).set_index("season")

        self.assertTrue(pd.isna(result.loc[2020, "availability_3yr"]))
        self.assertEqual(result.loc[2021, "availability_3yr"], 0.5)
        self.assertEqual(result.loc[2021, "availability_3yr_coverage"], 0.5)
        self.assertEqual(result.loc[2022, "availability_3yr"], 0.1875)
        self.assertEqual(result.loc[2022, "availability_3yr_coverage"], 0.8)
        self.assertEqual(result.loc[2023, "availability_3yr"], 0.6)
        self.assertEqual(result.loc[2023, "availability_3yr_coverage"], 1.0)


class MissingnessInvariantTests(unittest.TestCase):
    def test_observed_identity_and_workload_are_complete(self):
        frame = pd.read_csv("data/processed/training_data_v2.csv")

        self.assertFalse(
            frame[["age", "height_inches", "mpg", "availability_3yr"]]
            .isna().any().any()
        )
        self.assertNotIn("unresolved", set(frame["workload_source_status"]))

    def test_zero_game_rows_are_explicit_not_median_filled(self):
        frame = pd.read_csv("data/processed/training_data_v2.csv")
        missed = frame["workload_source_status"].eq("did_not_play")

        self.assertGreater(int(missed.sum()), 0)
        self.assertTrue(frame.loc[missed, "games"].eq(0).all())
        self.assertTrue(frame.loc[missed, "minutes"].eq(0).all())
        self.assertTrue(frame.loc[missed, "mpg"].eq(0).all())

    def test_bbref_fallback_has_all_seasons_and_unique_keys(self):
        frame = pd.read_csv("data/processed/advanced_stats.csv")

        self.assertEqual(set(frame["season"]), set(range(2019, 2027)))
        self.assertFalse(frame.duplicated(["player_url", "season"]).any())


if __name__ == "__main__":
    unittest.main()
