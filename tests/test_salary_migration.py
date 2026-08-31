"""Focused tests for Spotrac salary selection and dead-money exclusion."""

import numpy as np
import pandas as pd

import scripts.build_merged_salaries as merged
import scripts.check_caps as check_caps
import scripts.refresh_salaries as refresh
import src.features.build_dataset as dataset
import src.features.impact_identity as identity


def _row(**overrides):
    row = {
        "player_name_norm": "player",
        "season": 2022,
        "cap_hit": np.nan,
        "base": 10_000_000,
        "extra_amounts": "8000000",
        "statuses": "Active;Retained",
        "n_teams": 2,
        "split_total": 18_000_000,
        "contract_years": 2,
    }
    row.update(overrides)
    return row


def test_implied_cap_ignores_whole_dollar_rounding():
    rows = pd.DataFrame({"salary": [27_285_000.5]})
    assert check_caps._implied_cap(rows, 109_140_000) is None


def test_status_labels_exclude_retained_dead_money():
    value, branch = merged.resolve_row(_row(), set(), set())
    assert value == 10_000_000
    assert branch == "status_active"


def test_status_labels_can_select_second_active_amount():
    value, branch = merged.resolve_row(
        _row(base=7_000_000, extra_amounts="12000000",
             statuses="Retained;Active"),
        set(), set(),
    )
    assert value == 12_000_000
    assert branch == "status_active"


def test_retained_only_row_has_no_signed_salary():
    value, branch = merged.resolve_row(
        _row(statuses="Retained;Buyout"), set(), set()
    )
    assert np.isnan(value)
    assert branch == "retained_only"


def test_mismatched_status_count_falls_back_to_existing_rule():
    value, branch = merged.resolve_row(
        _row(extra_amounts=np.nan, statuses="Active;Retained", n_teams=1,
             split_total=np.nan),
        set(), set(),
    )
    assert value == 10_000_000
    assert branch == "single"


def test_covid_reduced_active_minimum_routes_to_cap_charge(monkeypatch):
    monkeypatch.setattr(merged, "PAGE_TRADED", {("kyle korver", 2019)})
    value, branch = merged.resolve_row(
        _row(
            player_name_norm="kyle korver",
            season=2019,
            base=2_404_456,
            extra_amounts="3440000",
            statuses="Active;Retained",
            n_teams=2,
            split_total=5_844_456,
            contract_years=1,
        ),
        set(),
        {("kyle korver", 2019)},
    )

    assert value == 1_620_564
    assert branch == "vet_min"


def test_non_covid_active_minimum_routes_to_existing_minimum_rule(monkeypatch):
    monkeypatch.setattr(merged, "PAGE_TRADED", {("rajon rondo", 2021)})
    value, branch = merged.resolve_row(
        _row(
            player_name_norm="rajon rondo",
            season=2021,
            base=1_487_859,
            extra_amounts="6012141",
            statuses="Active;Retained",
            n_teams=2,
            split_total=7_500_000,
            contract_years=1,
        ),
        set(),
        {("rajon rondo", 2021)},
    )

    assert value == 1_487_859
    assert branch == "vet_min"


def test_traded_components_that_sum_to_minimum_stay_vet_min(monkeypatch):
    monkeypatch.setattr(merged, "PAGE_TRADED", {("player", 2022)})
    value, branch = merged.resolve_row(
        _row(
            season=2022,
            base=1_000_000,
            extra_amounts="1721203",
            statuses="Active;Retained",
            n_teams=2,
            split_total=2_721_203,
            contract_years=1,
        ),
        set(),
        set(),
    )

    assert value == 1_836_090
    assert branch == "vet_min"


def test_unlabelled_active_minimum_does_not_replace_trade_sum(monkeypatch):
    monkeypatch.setattr(merged, "PAGE_TRADED", {("jordan nwora", 2022)})
    value, branch = merged.resolve_row(
        _row(
            player_name_norm="jordan nwora",
            season=2022,
            base=949_425,
            extra_amounts="1850575",
            statuses="Active;Retained",
            n_teams=2,
            split_total=2_800_000,
            contract_years=1,
        ),
        set(),
        set(),
    )

    assert value == 2_800_000
    assert branch == "status_active"


def test_salary_overlay_preserves_literal_nan_key(monkeypatch, tmp_path):
    pd.DataFrame([
        {
            "player_name_norm": "nan",
            "season": 2019,
            "salary": 708_426.5,
            "source": "spotrac",
            "branch": "status_active",
        }
    ]).to_csv(tmp_path / "merged_salaries.csv", index=False)
    monkeypatch.setattr(dataset, "PROCESSED_DIR", tmp_path)
    salaries = pd.DataFrame([
        {
            "player_name_norm": "nan",
            "season": 2019,
            "salary": 708_426,
        }
    ])

    out = dataset._apply_spotrac_salary_migration(salaries)

    assert len(out) == 1
    assert out.loc[0, "player_name_norm"] == "nan"
    assert out.loc[0, "salary"] == 708_426.5
    assert out.loc[0, "salary_branch"] == "status_active"


def test_retained_only_key_can_be_missing_from_identity_repairs(
    monkeypatch, tmp_path
):
    pd.DataFrame([
        {
            "player_name_norm": "timofey mozgov",
            "season": 2019,
            "salary": np.nan,
            "source": "spotrac",
            "branch": "retained_only",
        },
        {
            "player_name_norm": "active player",
            "season": 2019,
            "salary": 1_000_000,
            "source": "spotrac",
            "branch": "status_active",
        },
    ]).to_csv(tmp_path / "merged_salaries.csv", index=False)
    monkeypatch.setattr(dataset, "PROCESSED_DIR", tmp_path)
    salaries = pd.DataFrame([
        {"player_name_norm": "timofey mozgov", "season": 2019,
         "salary": 16_720_000},
        {"player_name_norm": "active player", "season": 2019,
         "salary": 1_000_000},
    ])

    migrated = dataset._apply_spotrac_salary_migration(salaries)
    dropped = migrated.attrs["salary_migration_dropped_keys"]

    corrections = tmp_path / "player_identity_corrections.csv"
    pd.DataFrame([
        {
            "player_name_norm": "timofey mozgov",
            "season": 2019,
            "games": 0,
            "source": "test",
            "note": "retained-only row",
        }
    ]).to_csv(corrections, index=False)
    monkeypatch.setattr(identity, "IDENTITY_CORRECTIONS", corrections)
    repaired = identity.apply_player_identity_corrections(
        migrated.assign(games=1), allowed_missing=dropped
    )

    assert set(zip(repaired["player_name_norm"], repaired["season"])) == {
        ("active player", 2019)
    }


def _base_salary_rows():
    return pd.DataFrame([
        {
            "player": "Missing Player",
            "player_url": "/players/m/missipl01.html",
            "team": "BOS",
            "season": 2025,
            "salary": 500_000,
            "age": 22,
            "source": "bbref_team_contracts",
            "team_name": pd.NA,
            "salary_cap": 154_647_000,
            "cap_pct": 500_000 / 154_647_000,
        }
    ])


def _spotrac_contracts():
    return pd.DataFrame([
        {
            "player_name_norm": "missing player",
            "season": 2026,
            "salary": 678_882,
            "teams": "PHX",
            "n_teams": 1,
            "table_kind": "contract",
        },
        {
            "player_name_norm": "historical player",
            "season": 2024,
            "salary": 1_000_000,
            "teams": "BOS",
            "n_teams": 1,
            "table_kind": "contract",
        },
        {
            "player_name_norm": "split player",
            "season": 2026,
            "salary": 2_000_000,
            "teams": "BOS;ATL",
            "n_teams": 2,
            "table_kind": "contract",
        },
    ])


def test_refresh_adds_only_current_single_team_spotrac_contract():
    identity_frame = pd.DataFrame([
        {
            "player": "Missing Player",
            "player_url": "/players/m/missipl01.html",
            "season": 2026,
            "age": 23,
        },
        {
            "player": "Historical Player",
            "player_url": "/players/h/histopl01.html",
            "season": 2024,
            "age": 25,
        },
        {
            "player": "Split Player",
            "player_url": "/players/s/splitpl01.html",
            "season": 2026,
            "age": 26,
        },
    ])

    out = refresh.add_spotrac_only_rows(
        _base_salary_rows(), _spotrac_contracts(), {2026}, identity_frame
    )

    added = out[(out["player"] == "Missing Player") & (out["season"] == 2026)]
    assert len(added) == 1
    assert added.iloc[0]["player_url"] == "/players/m/missipl01.html"
    assert added.iloc[0]["team"] == "PHO"
    assert added.iloc[0]["salary"] == 678_882
    assert added.iloc[0]["source"] == "spotrac_fa_backfill"
    assert not ((out["player"] == "Historical Player")
                & (out["season"] == 2024)).any()
    assert not ((out["player"] == "Split Player")
                & (out["season"] == 2026)).any()


def test_refresh_uses_fresh_one_year_fa_signing_when_player_page_is_stale():
    identity_frame = pd.DataFrame([{
        "player": "Fresh Signing",
        "player_url": "/players/f/freshsi01.html",
        "season": 2026,
        "age": 27,
    }])
    fa = pd.DataFrame([{
        "player_name_norm": "fresh signing",
        "season": 2026,
        "to_team": "GS",
        "aav_str": "$2,625,627",
        "contract_years": 1,
    }])

    out = refresh.add_spotrac_only_rows(
        _base_salary_rows(),
        _spotrac_contracts(),
        {2026},
        identity_frame,
        fa,
    )

    row = out[(out["player"] == "Fresh Signing") & (out["season"] == 2026)]
    assert len(row) == 1
    assert row.iloc[0]["salary"] == 2_625_627
    assert row.iloc[0]["team"] == "GSW"
