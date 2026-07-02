"""Shared scraping utilities: caching, rate limiting, fetching."""

import hashlib
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from config import CACHE_DIR, SCRAPE_DELAY_SECONDS

_last_request_time = 0.0


def _cache_path(url: str) -> Path:
    """Return a deterministic cache file path for a URL."""
    key = hashlib.sha256(url.encode()).hexdigest()[:16]
    safe_name = url.split("//")[-1].replace("/", "_").replace("?", "_")[:80]
    return CACHE_DIR / f"{safe_name}_{key}.html"


def fetch_html(url: str, *, force_refresh: bool = False, retries: int = 2) -> str:
    """Fetch a URL via headless Chromium, caching the result on disk.

    Returns the full page HTML after JavaScript has executed.
    Respects SCRAPE_DELAY_SECONDS between live requests.
    On 403, retries with exponential backoff before raising.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(url)

    if not force_refresh and path.exists():
        return path.read_text(encoding="utf-8")

    last_error = None
    for attempt in range(retries + 1):
        global _last_request_time
        elapsed = time.time() - _last_request_time
        wait = SCRAPE_DELAY_SECONDS * (2 ** attempt)
        if elapsed < wait:
            time.sleep(wait - elapsed)

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1920, "height": 1080},
            )
            page = context.new_page()
            resp = page.goto(url, timeout=30_000)
            _last_request_time = time.time()

            if resp.status == 403:
                browser.close()
                last_error = RuntimeError(
                    f"HTTP 403 for {url} — rate-limited (attempt {attempt + 1}/{retries + 1})"
                )
                if attempt < retries:
                    backoff = 30 * (2 ** attempt)
                    print(f"  Rate-limited, backing off {backoff}s...")
                    time.sleep(backoff)
                continue

            page.wait_for_load_state("domcontentloaded")
            time.sleep(3)
            html = page.content()
            browser.close()

            path.write_text(html, encoding="utf-8")
            return html

    raise last_error
