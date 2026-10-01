"""Tests for the Stage-1 P(max) term (v6.3.0)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.model.route_mixture import _tobit_beta_two_sided
from src.model.train import grabit_predict, max_term_z


class _ConstModel:
    """Stands in for the booster: the tree sum is a constant."""

    def predict(self, X):
        return np.full(len(X), 0.10)


class MaxTermTest(unittest.TestCase):
    def test_z_is_zero_above_the_ceiling(self):
        z = max_term_z(np.array([0.30, 0.30, 0.25]),
                       np.array([0.10, 0.35, 0.05]),
                       np.array([0.5, 0.9, 0.0]))
        np.testing.assert_allclose(z, [0.10, 0.0, 0.0])

    def test_predict_adds_the_term(self):
        X = pd.DataFrame({"kf_market_value": [0.10, 0.20]})
        frame = pd.DataFrame({"is_waived": [0, 0],
                              "prior_waiver_owed": [0, 0],
                              "max_eligible_pct": [0.30, 0.30]})
        results = {"waiver_beta": 0.0, "max_beta": 0.5}
        out = grabit_predict(_ConstModel(), results, X, frame,
                             p_max=np.array([1.0, 0.5]))
        np.testing.assert_allclose(out, [0.10 + 0.5 * 0.20,
                                         0.10 + 0.5 * 0.5 * 0.10])

    def test_predict_requires_p_max(self):
        X = pd.DataFrame({"kf_market_value": [0.10]})
        frame = pd.DataFrame({"is_waived": [0], "prior_waiver_owed": [0],
                              "max_eligible_pct": [0.30]})
        with self.assertRaises(ValueError):
            grabit_predict(_ConstModel(), {"max_beta": 0.5}, X, frame)

    def test_two_sided_tobit_recovers_beta(self):
        rng = np.random.default_rng(0)
        n = 4000
        base = rng.uniform(0.02, 0.20, n)
        z = rng.uniform(0.0, 0.15, n)
        latent = base + 0.6 * z + rng.normal(0, 0.01, n)
        ceiling, floor = 0.25, np.full(n, 0.03)
        right = latent >= ceiling
        left = latent <= floor
        y = np.clip(latent, floor, ceiling)
        beta = _tobit_beta_two_sided(y, base, z, right, left, floor,
                                     (0.0, 1.0), x0=0.3)
        self.assertAlmostEqual(beta, 0.6, delta=0.03)


if __name__ == "__main__":
    unittest.main()
