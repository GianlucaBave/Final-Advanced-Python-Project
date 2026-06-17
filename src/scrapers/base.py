"""Abstract base scraper (S1 — OOP; S4 — exceptions/backoff).

``BaseScraper`` encapsulates everything the concrete scrapers share:

* a configured :class:`requests.Session` with a realistic ``User-Agent``,
* polite, fixed-rate request pacing,
* exponential backoff with retry on transient (429 / 5xx) failures,
* a uniform :meth:`scrape` template method returning ``list[Listing]``.

Subclasses implement :meth:`_search` (one network round-trip → raw records) and
:meth:`_parse` (raw record → :class:`Listing`). The public :meth:`scrape`
orchestrates them with error handling so a single bad page never aborts a run.
"""

from __future__ import annotations

import abc
import time
from datetime import datetime, timezone
from typing import Any, Iterable

import requests

from ..exceptions import ScraperError
from ..logging_setup import get_logger
from ..schema import Listing


class BaseScraper(abc.ABC):
    """Template for all source scrapers."""

    #: Subclasses set this; used for logging and the ``Listing.source`` field.
    source_name: str = "base"

    def __init__(
        self,
        user_agent: str,
        rate_limit_seconds: float = 1.0,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        timeout: float = 20.0,
        extra_headers: dict[str, str] | None = None,
        impersonate: str | None = None,
        warm_url: str | None = None,
    ) -> None:
        self.log = get_logger(f"scraper.{self.source_name}")
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.timeout = timeout
        self._last_request_ts = 0.0
        # Sites behind Akamai/PerimeterX 403 a "cold" search request but allow it
        # once a browsing session exists. Fetching this URL once seeds the
        # cookies that get subsequent searches through the bot wall.
        self.warm_url = warm_url
        self._warmed = False

        headers = {
            "User-Agent": user_agent,
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        }
        if extra_headers:
            headers.update(extra_headers)

        # Transport selection: sites behind TLS-fingerprint bot protection
        # (eBay's Akamai) reject vanilla `requests` regardless of headers. When
        # ``impersonate`` is set we use curl_cffi, which mimics a real Chrome TLS
        # (JA3) handshake and gets through. Falls back to requests if curl_cffi
        # is unavailable.
        self._impersonate = impersonate
        self._net_errors: tuple = (requests.RequestException,)
        if impersonate:
            try:
                from curl_cffi import requests as creq  # type: ignore

                self.session = creq.Session(impersonate=impersonate, headers=headers)
                self._net_errors = (Exception,)
                self.log.debug("transport: curl_cffi (impersonate=%s)", impersonate)
            except ImportError:
                self.log.warning(
                    "curl_cffi not installed — falling back to requests "
                    "(this source may be blocked by TLS fingerprinting)"
                )
                self.session = requests.Session()
                self.session.headers.update(headers)
        else:
            self.session = requests.Session()
            self.session.headers.update(headers)

    # --- shared HTTP machinery -------------------------------------------
    def _throttle(self) -> None:
        """Sleep just long enough to honour the per-source rate limit."""
        elapsed = time.monotonic() - self._last_request_ts
        wait = self.rate_limit_seconds - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_ts = time.monotonic()

    def _request(self, url: str, *, params: dict | None = None) -> requests.Response:
        """GET with throttling and exponential backoff on transient errors.

        Raises :class:`ScraperError` only after ``max_retries`` exhausted on a
        retryable status, or immediately on a non-retryable client error.
        """
        self._ensure_warmed()
        last_exc: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except self._net_errors as exc:  # network-level failure
                last_exc = exc
                sleep_s = self.backoff_base ** attempt
                self.log.warning(
                    "network error (%s) attempt %d/%d — backing off %.1fs",
                    type(exc).__name__, attempt, self.max_retries, sleep_s,
                )
                time.sleep(sleep_s)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = self.backoff_base ** attempt
                self.log.warning(
                    "HTTP %d on attempt %d/%d — backing off %.1fs",
                    resp.status_code, attempt, self.max_retries, sleep_s,
                )
                time.sleep(sleep_s)
                continue

            if resp.status_code >= 400:
                # Non-retryable client error (403/404/...) — fail fast.
                raise ScraperError(
                    f"{self.source_name}: HTTP {resp.status_code} for {url}"
                )
            return resp

        raise ScraperError(
            f"{self.source_name}: exhausted {self.max_retries} retries for {url}"
            + (f" (last error: {last_exc})" if last_exc else "")
        )

    def _ensure_warmed(self) -> None:
        """Fetch ``warm_url`` once to seed session cookies (best-effort)."""
        if not self.warm_url or self._warmed:
            return
        self._warmed = True  # set first so a failure never loops
        try:
            self._throttle()
            self.session.get(self.warm_url, timeout=self.timeout)
            self.log.debug("warmed session via %s", self.warm_url)
        except Exception as exc:  # noqa: BLE001 — warming is best-effort
            self.log.debug("session warm failed (%s) — continuing cold", exc)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    # --- template method --------------------------------------------------
    @abc.abstractmethod
    def _search(self, model: str, city: str, **kwargs: Any) -> Iterable[dict]:
        """Yield raw source-native records for one (model, city) query."""

    @abc.abstractmethod
    def _parse(self, record: dict, *, model: str, city: str) -> Listing | None:
        """Convert one raw record into a :class:`Listing` (or ``None`` to skip)."""

    def scrape(self, model: str, city: str, **kwargs: Any) -> list[Listing]:
        """Public entry point: search + parse one (model, city), error-safe.

        A failure in a single record is logged and skipped; a failure of the
        whole search is logged and returns an empty list so the caller's sweep
        continues. Hard configuration errors still propagate.
        """
        listings: list[Listing] = []
        try:
            records = list(self._search(model, city, **kwargs))
        except ScraperError as exc:
            self.log.error("search failed for %s / %s: %s", model, city, exc)
            return listings

        for rec in records:
            try:
                listing = self._parse(rec, model=model, city=city)
            except Exception as exc:  # noqa: BLE001 — never let one row kill a run
                self.log.debug("parse error (skipped): %s", exc)
                continue
            if listing is not None:
                listings.append(listing)

        self.log.debug("scraped %d listings for %s / %s", len(listings), model, city)
        return listings
