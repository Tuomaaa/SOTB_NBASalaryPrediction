"""Tests for the in-season KF re-pricing anchors (v6.0.4)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.features.kf_market_value import build_anchor_map, load_reprice_events


def _frames():
    ev = pd.DataFrame([
        {"player_name_norm": "p", "season": 2019, "cap_pct": 0.20,
         "floor_pct": 0.015, "year_in_contract": 1},
        {"player_name_norm": "p", "season": 2021, "cap_pct": 0.015,
         "floor_pct": 0.015, "year_in_contract": 1},
    ])
    full = ev.copy()
    return ev, full


class RepriceAnchorTests(unittest.TestCase):
    def test_default_keeps_the_old_contract(self):
        ev, full = _frames()
        _, _, tier, anchor = build_anchor_map(ev, full, expand_anchors=False)
        self.assertEqual(tier[1], 1)
        self.assertAlmostEqual(anchor[1], 0.20)

    def test_floor_event_overrides_and_uses_the_floor(self):
        ev, full = _frames()
        events = {"p": {2020: (pd.Timestamp("2021-03-01"), "floor", None)}}
        _, _, tier, anchor = build_anchor_map(
            ev, full, expand_anchors=False, reprice_events=events)
        self.assertEqual(tier[1], 1)
        self.assertAlmostEqual(anchor[1], 0.015)

    def test_zero_event_is_weak(self):
        ev, full = _frames()
        events = {"p": {2020: (pd.Timestamp("2021-01-10"), "zero", None)}}
        _, _, tier, anchor = build_anchor_map(
            ev, full, expand_anchors=False, reprice_events=events)
        self.assertEqual(tier[1], 2)
        self.assertEqual(anchor[1], 0.0)

    def test_price_event_is_floored_at_the_minimum(self):
        ev, full = _frames()
        events = {"p": {2020: (pd.Timestamp("2021-02-01"), "price", 0.005)}}
        _, _, _, anchor = build_anchor_map(
            ev, full, expand_anchors=False, reprice_events=events)
        self.assertAlmostEqual(anchor[1], 0.015)

    def test_late_june_signing_is_not_a_prior_season_event(self):
        """A June 29 deal starts the next season; it must not anchor itself."""
        ev = load_reprice_events()
        self.assertNotIn(2023, ev.get("patrick williams", {}))
        self.assertNotIn(2021, ev.get("thaddeus young", {}))

    def test_rest_of_season_after_a_buyout_is_a_floor_event(self):
        ev = load_reprice_events()
        self.assertEqual(ev["andre drummond"][2020][1], "floor")


if __name__ == "__main__":
    unittest.main()
