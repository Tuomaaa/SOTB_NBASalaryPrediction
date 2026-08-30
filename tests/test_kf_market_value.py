import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.features.kf_market_value import (
    build_anchor_map,
    compute_kf_column,
    kalman_update,
)


class KFMarketEventSeparationTests(unittest.TestCase):
    def setUp(self):
        self.full = pd.DataFrame({
            "player_name_norm": ["test player"] * 4,
            "season": [2020, 2021, 2022, 2023],
            "cap_pct": [0.10, 0.12, 0.20, 0.22],
            "prev_cap_pct": [0.08, 0.10, 0.12, 0.20],
            "floor_pct": [0.01] * 4,
            "year_in_contract": [1, 2, 1, 2],
        })
        self.events = self.full.iloc[[0, 2]].copy()
        self.target = self.full.iloc[[3]].copy()

    @patch("src.features.kf_market_value._load_debut_seasons",
           return_value={})
    @patch("src.features.kf_market_value._load_rookie_scale_set",
           return_value=set())
    def test_inference_target_uses_historical_market_frame(self, *_):
        values = compute_kf_column(
            self.target, self.full, lambda rows: np.full(len(rows), 0.30),
            r_var=0.01, players=set(), expand_anchors=False,
            market_events=self.events)

        self.assertEqual(values[0], 0.20)

    @patch("src.features.kf_market_value._load_debut_seasons",
           return_value={})
    @patch("src.features.kf_market_value._load_rookie_scale_set",
           return_value=set())
    def test_current_and_future_events_never_anchor_target(self, *_):
        events = pd.concat([
            self.events,
            pd.DataFrame({
                "player_name_norm": ["test player", "test player"],
                "season": [2023, 2024],
                "cap_pct": [0.90, 0.95],
                "prev_cap_pct": [0.20, 0.90],
                "floor_pct": [0.01, 0.01],
                "year_in_contract": [1, 1],
            }),
        ], ignore_index=True)
        _, _, tier, anchor = build_anchor_map(
            self.target, self.full, expand_anchors=False,
            market_events=events)

        self.assertEqual(tier[0], 1)
        self.assertEqual(anchor[0], 0.20)

    @patch("src.features.kf_market_value._load_debut_seasons",
           return_value={})
    @patch("src.features.kf_market_value._load_rookie_scale_set",
           return_value=set())
    def test_explicit_same_event_frame_is_backward_compatible(self, *_):
        predict = lambda rows: np.full(len(rows), 0.30)
        old = compute_kf_column(
            self.events, self.full, predict, r_var=0.01,
            players=set(), expand_anchors=False)
        explicit = compute_kf_column(
            self.events, self.full, predict, r_var=0.01,
            players=set(), expand_anchors=False,
            market_events=self.events)

        np.testing.assert_array_equal(old, explicit)

    @patch("src.features.kf_market_value._load_debut_seasons",
           return_value={})
    @patch("src.features.kf_market_value._load_rookie_scale_set",
           return_value={
               ("rookie player", 2021),
               ("rookie player", 2022),
               ("rookie player", 2023),
           })
    def test_rookie_target_uses_intermediate_model_measurement(self, *_):
        full = pd.DataFrame({
            "player_name_norm": ["rookie player"] * 3,
            "season": [2021, 2022, 2023],
            "cap_pct": [0.05, 0.052, 0.054],
            "prev_cap_pct": [0.04, 0.05, 0.052],
            "floor_pct": [0.01] * 3,
            "year_in_contract": [1, 2, 3],
        })
        target = full.iloc[[2]].copy()
        events = pd.DataFrame({
            "player_name_norm": ["other player"],
            "season": [2022],
            "cap_pct": [0.10],
            "prev_cap_pct": [0.08],
            "floor_pct": [0.01],
            "year_in_contract": [1],
        })

        values = compute_kf_column(
            target, full, lambda rows: np.full(len(rows), 0.30),
            r_var=0.01, players=set(), expand_anchors=False,
            market_events=events)
        expected = kalman_update(0.05, [0.30], q=0.04, r=0.01, p0=0.01)

        self.assertAlmostEqual(values[0], expected)
        self.assertGreater(values[0], target["prev_cap_pct"].iloc[0])


if __name__ == "__main__":
    unittest.main()
