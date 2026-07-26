"""Regression tests for the previous-waiver feature."""

import unittest

import numpy as np
import pandas as pd

from src.features.waiver_history import (
    attach_waiver_history,
    classify_transaction,
)


class WaiverHistoryTests(unittest.TestCase):
    def test_classifies_spotrac_waiver_wording(self):
        self.assertEqual(
            classify_transaction("Waived by Charlotte (CHA) - agreed to buyout"),
            "waived",
        )
        self.assertEqual(
            classify_transaction(
                "Waived by Charlotte (CHA) via buyout; leaving behind dead cap"
            ),
            "waived",
        )
        self.assertEqual(
            classify_transaction("Signed a 1 year contract with Philadelphia"),
            "signed",
        )

    def test_uses_only_waivers_before_signing_within_365_days(self):
        frame = pd.DataFrame([
            {"player_name_norm": "positive", "season": 2023, "salary": 3_200_000},
            {"player_name_norm": "negative", "season": 2023, "salary": 3_200_000},
            {"player_name_norm": "unknown", "season": 2023, "salary": 3_200_000},
        ])
        signings = pd.DataFrame([
            {
                "player_name_norm": player,
                "signing_date": "2023-10-04",
                "signing_season": 2023,
                "contract_years": 1,
                "total_value": 3_200_000,
                "is_extension": 0,
                "contract_class": "standard",
                "team": "HOU",
                "fa_year_matched": 2024,
                "match_confidence": "exact",
                "tx_text": "Signed a 1 year $3.2 million contract",
            }
            for player in ("positive", "negative")
        ])
        events = pd.DataFrame([
            {
                "player_name_norm": "positive",
                "transaction_date": "2023-09-30",
                "event_type": "waived",
                "tx_text": "Waived by San Antonio (SAS)",
            },
            {
                "player_name_norm": "positive",
                "transaction_date": "2023-10-05",
                "event_type": "waived",
                "tx_text": "Future waiver must not leak",
            },
            {
                "player_name_norm": "negative",
                "transaction_date": "2022-09-01",
                "event_type": "waived",
                "tx_text": "Outside the registered lookback",
            },
        ])

        result = attach_waiver_history(frame, events, signings)

        self.assertEqual(result.loc[0, "is_waived"], 1.0)
        self.assertEqual(result.loc[1, "is_waived"], 0.0)
        self.assertTrue(np.isnan(result.loc[2, "is_waived"]))
        self.assertEqual(result["is_waived_known"].tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(
            result.loc[0, "prior_waiver_text"],
            "Waived by San Antonio (SAS)",
        )

    def test_falls_back_to_unpriced_same_season_signing(self):
        frame = pd.DataFrame([
            {"player_name_norm": "fallback", "season": 2020, "salary": 1_600_000},
        ])
        signings = pd.DataFrame([{
            "player_name_norm": "fallback",
            "signing_date": "2020-12-01",
            "signing_season": 2020,
            "contract_years": np.nan,
            "total_value": np.nan,
            "is_extension": 0,
            "contract_class": "standard",
            "team": "LAC",
            "fa_year_matched": np.nan,
            "match_confidence": "unmatchable",
            "tx_text": "Signed a contract with Los Angeles (LAC)",
        }])
        events = pd.DataFrame([{
            "player_name_norm": "fallback",
            "transaction_date": "2020-02-18",
            "event_type": "waived",
            "tx_text": "Waived by Detroit (DET) - agree to buyout",
        }])

        result = attach_waiver_history(frame, events, signings)

        self.assertEqual(result.loc[0, "is_waived"], 1.0)
        self.assertEqual(result.loc[0, "is_waived_known"], 1.0)


if __name__ == "__main__":
    unittest.main()
