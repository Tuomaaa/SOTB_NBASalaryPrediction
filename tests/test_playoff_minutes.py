"""Regression tests for the playoff-minutes feature and the classifier's list.

Two separate contracts land together in v8.6x and are tested together because
one exists to protect the other:

  - `playoff_mpg_diff` = po_mpg - mpg, zero where the player did not appear;
  - the route classifier's feature list is CLF_BASE_COLS + CLF_EXTRA_COLS and
    does NOT follow FEATURE_COLS (v8.6x). A 17th regression feature must leave
    the classifier's 36 columns untouched, because P(max) feeds the knife-edge
    Stage-2 push gate.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.scrape_playoff_mpg import _normalize_name
from src.features.playoff_minutes import attach_playoff_mpg
from src.model import train as train_mod
from src.model.route_mixture import (
    CLF_BASE_COLS, CLF_EXTRA_COLS, attach_clf_features,
)


class PlayoffMinutesTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "playoff_mpg.csv"
        pd.DataFrame([
            {"player_name": "Played Deep", "player_name_norm": "played deep",
             "season": 2024, "po_team": "BOS", "po_games": 19, "po_mpg": 38.0},
            {"player_name": "Benched Guy", "player_name_norm": "benched guy",
             "season": 2024, "po_team": "BOS", "po_games": 4, "po_mpg": 6.0},
            {"player_name": "Enes Kanter", "player_name_norm": "enes kanter",
             "season": 2021, "po_team": "POR", "po_games": 6, "po_mpg": 10.0},
        ]).to_csv(self.path, index=False)

    def tearDown(self):
        self._tmp.cleanup()

    def _frame(self):
        return pd.DataFrame([
            {"player_name_norm": "played deep", "season": 2024, "mpg": 33.0},
            {"player_name_norm": "benched guy", "season": 2024, "mpg": 24.0},
            {"player_name_norm": "no playoffs", "season": 2024, "mpg": 20.0},
            {"player_name_norm": "enes freedom", "season": 2021, "mpg": 24.0},
        ])

    def test_difference_and_zero_fill(self):
        out = attach_playoff_mpg(self._frame(), path=self.path)
        self.assertAlmostEqual(out.loc[0, "playoff_mpg_diff"], 5.0)
        self.assertAlmostEqual(out.loc[1, "playoff_mpg_diff"], -18.0)
        # No playoff appearance is a fact about the team, not a missing value.
        self.assertEqual(out.loc[2, "playoff_mpg_diff"], 0.0)
        self.assertTrue(np.isnan(out.loc[2, "po_mpg"]))
        self.assertFalse(out["playoff_mpg_diff"].isna().any())

    def test_alias_resolves_a_renamed_player(self):
        out = attach_playoff_mpg(self._frame(), path=self.path)
        self.assertAlmostEqual(out.loc[3, "playoff_mpg_diff"], -14.0)

    def test_row_count_and_idempotence(self):
        frame = self._frame()
        once = attach_playoff_mpg(frame, path=self.path)
        twice = attach_playoff_mpg(once, path=self.path)
        self.assertEqual(len(twice), len(frame))
        self.assertEqual(sum(c == "po_mpg" for c in twice.columns), 1)
        pd.testing.assert_series_equal(once["playoff_mpg_diff"],
                                       twice["playoff_mpg_diff"])

    def test_hall_of_fame_asterisk_is_stripped(self):
        # Without this, Dwight Howard 2020/2021 silently read as "did not play".
        self.assertEqual(_normalize_name("Dwight Howard*"), "dwight howard")
        self.assertEqual(_normalize_name("Nikola Jokić"), "nikola jokic")


class ClassifierFeatureListTests(unittest.TestCase):
    """v8.6x: the classifier's inputs are curated, not inherited."""

    def setUp(self):
        self._saved = list(train_mod.FEATURE_COLS)

    def tearDown(self):
        train_mod.FEATURE_COLS = self._saved

    def _frame(self):
        df = pd.DataFrame({"player_name_norm": ["nobody a", "nobody b"],
                           "season": [2024, 2025]})
        for col in CLF_BASE_COLS:
            df[col] = 0.0
        return df

    def test_list_is_base_plus_extras(self):
        _, clf = attach_clf_features(self._frame())
        self.assertEqual(clf, list(CLF_BASE_COLS) + list(CLF_EXTRA_COLS))
        self.assertEqual(len(clf), 36)

    def test_growing_feature_cols_does_not_grow_the_classifier(self):
        _, before = attach_clf_features(self._frame())
        train_mod.FEATURE_COLS = list(train_mod.FEATURE_COLS) + ["a_new_feature"]
        _, after = attach_clf_features(self._frame())
        self.assertEqual(before, after)
        self.assertNotIn("a_new_feature", after)

    def test_missing_base_column_is_an_error_not_a_silent_drop(self):
        frame = self._frame().drop(columns=["mpg"])
        with self.assertRaises(KeyError):
            attach_clf_features(frame)


if __name__ == "__main__":
    unittest.main()
