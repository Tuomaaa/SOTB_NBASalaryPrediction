"""Regression tests for the previous-waiver feature."""

import unittest

import numpy as np
import pandas as pd

from src.features.waiver_history import (
    _resolve_waiver_no_signing,
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


    def test_fallback_uses_first_signing_of_season(self):
        """An in-season waiver must not leak through a later 10-day deal."""
        frame = pd.DataFrame([
            {"player_name_norm": "journeyman", "season": 2023,
             "salary": 2_000_000},
        ])
        base = {
            "player_name_norm": "journeyman", "signing_season": 2023,
            "contract_years": np.nan, "total_value": np.nan,
            "is_extension": 0, "team": "NYK", "fa_year_matched": np.nan,
            "match_confidence": "unmatchable",
        }
        signings = pd.DataFrame([
            {**base, "signing_date": "2023-09-15",
             "contract_class": "standard", "tx_text": "Signed a contract"},
            {**base, "signing_date": "2024-01-30",
             "contract_class": "10-day", "tx_text": "Signed a 10-day contract"},
        ])
        events = pd.DataFrame([{
            "player_name_norm": "journeyman",
            "transaction_date": "2024-01-07",
            "event_type": "waived",
            "tx_text": "Waived by Washington (WAS)",
        }])

        result = attach_waiver_history(frame, events, signings)

        self.assertEqual(result.loc[0, "is_waived"], 0.0)
        self.assertEqual(result.loc[0, "is_waived_known"], 1.0)

    def test_fallback_skips_camp_deal_cut_before_opener(self):
        """A camp cut did not price the row; its waiver is prior information."""
        frame = pd.DataFrame([
            {"player_name_norm": "camp", "season": 2025, "salary": 1_755_198},
        ])
        base = {
            "player_name_norm": "camp", "signing_season": 2025,
            "contract_years": np.nan, "total_value": np.nan,
            "is_extension": 0, "team": "GSW", "fa_year_matched": np.nan,
            "match_confidence": "unmatchable",
        }
        signings = pd.DataFrame([
            {**base, "signing_date": "2025-10-01",
             "contract_class": "standard", "tx_text": "Exhibit 9"},
            {**base, "signing_date": "2025-12-01",
             "contract_class": "rest-of-season", "tx_text": "Rest-of-Season"},
        ])
        events = pd.DataFrame([{
            "player_name_norm": "camp",
            "transaction_date": "2025-10-18",
            "event_type": "waived",
            "tx_text": "Waived by Golden State (GSW)",
        }])

        result = attach_waiver_history(frame, events, signings)

        self.assertEqual(result.loc[0, "is_waived"], 1.0)

    def test_resolve_no_signing_no_waivers(self):
        """Player with transaction page but zero waiver events -> not waived."""
        player_tx = pd.DataFrame([{
            "player_name_norm": "clean",
            "transaction_date": pd.Timestamp("2022-07-01"),
            "event_type": "signed",
            "tx_text": "Signed a 2 year contract",
        }])
        result = _resolve_waiver_no_signing(player_tx, season=2023)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 0.0)

    def test_resolve_no_signing_waiver_outside_window(self):
        """Waiver is clearly outside any possible lookback -> not waived."""
        player_tx = pd.DataFrame([{
            "player_name_norm": "old_waiver",
            "transaction_date": pd.Timestamp("2018-01-15"),
            "event_type": "waived",
            "tx_text": "Waived by Phoenix (PHX)",
        }])
        result = _resolve_waiver_no_signing(player_tx, season=2023)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 0.0)

    def test_resolve_no_signing_waiver_in_tight_window(self):
        """Waiver clearly inside every possible lookback -> waived."""
        player_tx = pd.DataFrame([{
            "player_name_norm": "recent_waiver",
            "transaction_date": pd.Timestamp("2022-03-01"),
            "event_type": "waived",
            "tx_text": "Waived by Brooklyn (BRK)",
        }])
        result = _resolve_waiver_no_signing(player_tx, season=2023)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], 1.0)
        self.assertEqual(result[2], "Waived by Brooklyn (BRK)")

    def test_resolve_no_signing_waiver_ambiguous(self):
        """Waiver between tight and wide window -> None (leave unknown)."""
        # Season 2023: wide window July 2021 - Oct 2022
        # Tight window: Oct 2021 - July 2022
        # A waiver in Aug 2022 is in wide but not tight -> ambiguous
        player_tx = pd.DataFrame([{
            "player_name_norm": "ambiguous",
            "transaction_date": pd.Timestamp("2022-08-15"),
            "event_type": "waived",
            "tx_text": "Waived by Houston (HOU)",
        }])
        result = _resolve_waiver_no_signing(player_tx, season=2023)
        self.assertIsNone(result)

    def test_conservative_resolution_via_attach(self):
        """attach_waiver_history resolves rows with tx data but no signing."""
        frame = pd.DataFrame([
            {"player_name_norm": "has_tx_no_sd", "season": 2023, "salary": 2_000_000},
        ])
        # Player has transaction events (putting them in page_coverage)
        # but no signing date -> _resolve_waiver_no_signing kicks in
        events = pd.DataFrame([{
            "player_name_norm": "has_tx_no_sd",
            "transaction_date": "2023-06-01",
            "event_type": "signed",
            "tx_text": "Signed a contract with Miami (MIA)",
        }])
        # Empty signing dates -> no match
        signings = pd.DataFrame(columns=[
            "player_name_norm", "signing_date", "signing_season",
            "contract_years", "total_value", "is_extension",
            "contract_class", "team", "fa_year_matched",
            "match_confidence", "tx_text",
        ])

        result = attach_waiver_history(frame, events, signings)
        # No waiver events -> definitively not waived
        self.assertEqual(result.loc[0, "is_waived"], 0.0)
        self.assertEqual(result.loc[0, "is_waived_known"], 1.0)


if __name__ == "__main__":
    unittest.main()
