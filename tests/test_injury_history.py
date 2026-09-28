import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.features.injury_history import (
    classify_injury_reason,
    injury_score_as_of,
    parse_injury_page,
)


HTML = """
<div class="tab-pane fade" id="injuries" role="tabpanel">
<table><tbody>
<tr><td>2024-25</td><td>MIL</td><td>Mar 20, 2025 - Apr 13, 2025</td><td>14</td><td>Groin</td></tr>
<tr><td>2025-26</td><td>POR</td><td>Oct 22, 2025 - Apr 12, 2026</td><td>82</td><td>Achilles</td></tr>
<tr><td>2024-25</td><td>MIL</td><td>Feb 01, 2025 - Feb 01, 2025</td><td>1</td><td>Rest</td></tr>
</tbody></table></div><div class="tab-pane fade" id="transactions">
"""


class InjuryHistoryTest(unittest.TestCase):
    def test_parser_and_type_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "player.html"
            path.write_text(HTML, encoding="utf-8")
            rows = parse_injury_page(path, "test player")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows.iloc[0]["games_missed"], 14)
        self.assertEqual(rows.iloc[1]["injury_category"], "severe_structural")
        self.assertEqual(classify_injury_reason("Rest"), ("non_injury_dnp", 0.0))

    def test_cutoff_excludes_future_start_and_non_injury(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "player.html"
            path.write_text(HTML, encoding="utf-8")
            rows = parse_injury_page(path, "test player")
        score, count, games = injury_score_as_of(rows, "2025-07-19")
        self.assertGreater(score, 0)
        self.assertEqual(count, 1)
        self.assertEqual(games, 14)

    def test_open_event_does_not_leak_eventual_games_total(self):
        rows = pd.DataFrame([{
            "start_date": pd.Timestamp("2025-01-01"),
            "end_date": pd.Timestamp("2025-06-01"),
            "games_missed": 50,
            "injury_category": "severe_structural",
            "type_weight": 1.5,
        }])
        _, count, games = injury_score_as_of(rows, "2025-02-01")
        self.assertEqual(count, 1)
        self.assertEqual(games, 1)


if __name__ == "__main__":
    unittest.main()
