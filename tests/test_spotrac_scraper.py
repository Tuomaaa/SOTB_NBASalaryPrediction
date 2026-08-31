"""Regression tests for the guarded Spotrac player-page fetch path."""

from unittest.mock import Mock

import scripts.scrape_spotrac_players as scraper


def _html(canonical: str) -> str:
    return ("<html><head><link rel='canonical' href='" + canonical
            + "'></head></html>")


def _configure(monkeypatch, tmp_path, responses) -> Mock:
    get = Mock(side_effect=responses)
    monkeypatch.setattr(scraper, "PLAYER_CACHE", tmp_path)
    monkeypatch.setattr(scraper, "SCRAPE_DELAY_SECONDS", 0)
    monkeypatch.setattr(scraper, "_LAST_SPOTRAC_REQUEST_AT", None)
    monkeypatch.setattr(scraper.requests, "get", get)
    return get


def test_valid_cache_is_reused_without_a_request(monkeypatch, tmp_path):
    cached = tmp_path / "player.html"
    cached.write_text(
        _html("https://www.spotrac.com/nba/player/_/id/1/player"),
        encoding="utf-8",
    )
    get = _configure(monkeypatch, tmp_path, [])

    assert scraper.scrape_player("https://example.invalid", "player") == cached
    get.assert_not_called()


def test_invalid_cache_is_deleted_and_refetched(monkeypatch, tmp_path):
    cached = tmp_path / "player.html"
    cached.write_text(
        _html("https://www.spotrac.com/nfl/player/_/id/1/player"),
        encoding="utf-8",
    )
    response = Mock(
        status_code=200,
        text=_html("https://www.spotrac.com/nba/player/_/id/1/player"),
    )
    get = _configure(monkeypatch, tmp_path, [response])

    assert scraper.scrape_player("https://spotrac.test/player", "player") == cached
    assert scraper.page_defect(cached) is None
    assert get.call_count == 1


def test_throttling_statuses_retry_until_valid(monkeypatch, tmp_path):
    responses = [
        Mock(status_code=403, text=""),
        Mock(status_code=429, text=""),
        Mock(
            status_code=200,
            text=_html("https://www.spotrac.com/nba/player/_/id/1/player"),
        ),
    ]
    get = _configure(monkeypatch, tmp_path, responses)

    result = scraper.scrape_player("https://spotrac.test/player", "player")

    assert result == tmp_path / "player.html"
    assert get.call_count == 3


def test_two_callers_cannot_burst_requests(monkeypatch, tmp_path):
    responses = [
        Mock(
            status_code=200,
            text=_html("https://www.spotrac.com/nba/player/_/id/1/one"),
        ),
        Mock(
            status_code=200,
            text=_html("https://www.spotrac.com/nba/player/_/id/2/two"),
        ),
    ]
    get = _configure(monkeypatch, tmp_path, responses)
    monkeypatch.setattr(scraper, "SCRAPE_DELAY_SECONDS", 3)
    monotonic = Mock(side_effect=[10, 10, 11, 14])
    sleep = Mock()
    monkeypatch.setattr(scraper.time, "monotonic", monotonic)
    monkeypatch.setattr(scraper.time, "sleep", sleep)

    assert scraper.scrape_player("https://spotrac.test/one", "one")
    assert scraper.scrape_player("https://spotrac.test/two", "two")

    assert get.call_count == 2
    sleep.assert_called_once_with(2)


def test_invalid_response_page_is_removed_before_retry(monkeypatch, tmp_path):
    responses = [
        Mock(
            status_code=200,
            text=_html("https://www.spotrac.com/nfl/player/_/id/1/player"),
        ),
        Mock(
            status_code=200,
            text=_html("https://www.spotrac.com/nba/player/_/id/1/player"),
        ),
    ]
    get = _configure(monkeypatch, tmp_path, responses)

    result = scraper.scrape_player("https://spotrac.test/player", "player")

    assert result == tmp_path / "player.html"
    assert scraper.page_defect(result) is None
    assert get.call_count == 2


def test_valid_nba_page_for_wrong_player_is_rejected(tmp_path):
    cached = tmp_path / "mouhamadou-gueye.html"
    cached.write_text(
        _html(
            "https://www.spotrac.com/nba/player/_/id/84468/mouhamed-gueye"
        ),
        encoding="utf-8",
    )

    defect = scraper.page_defect(
        cached,
        expected_url=(
            "https://www.spotrac.com/nba/player/_/id/79540/"
            "mouhamadou-gueye"
        ),
    )

    assert "player id does not match" in defect
