"""Tests for the Stage-1 waiver term (v6.2.0: money-owed waivers only)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.model.train import grabit_predict, waiver_z


def _frame():
    return pd.DataFrame({
        "is_waived": [1, 1, 0, np.nan, 1],
        "prior_waiver_owed": [1, 0, 1, np.nan, np.nan],
        "kf_market_value": [0.20, 0.10, 0.30, 0.05, 0.08],
    })


class _TreesOnly:
    """Stands in for a fitted model: predict returns a constant."""

    def predict(self, X):
        return np.full(len(X), 0.05)


class WaiverTermTest(unittest.TestCase):
    def test_all_waived(self):
        np.testing.assert_allclose(waiver_z(_frame()),
                                   [0.20, 0.10, 0.0, 0.0, 0.08])

    def test_owed_only(self):
        np.testing.assert_allclose(waiver_z(_frame(), owed_only=True),
                                   [0.20, 0.0, 0.0, 0.0, 0.0])

    def test_predict_adds_owed_term_on_filled_kf(self):
        frame = _frame()
        X = pd.DataFrame({"kf_market_value": [0.25, 0.10, 0.30, 0.05, 0.08]})
        latent = grabit_predict(_TreesOnly(), {"waiver_beta": -0.5}, X, frame)
        np.testing.assert_allclose(latent, [0.05 - 0.125, 0.05, 0.05, 0.05, 0.05])

    def test_zero_beta_is_trees_only(self):
        X = pd.DataFrame({"kf_market_value": [0.2] * 5})
        latent = grabit_predict(_TreesOnly(), {}, X, _frame())
        np.testing.assert_allclose(latent, np.full(5, 0.05))


if __name__ == "__main__":
    unittest.main()
