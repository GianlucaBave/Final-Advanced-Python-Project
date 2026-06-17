"""eBay scraper — SOLD/COMPLETED listings via HTML, no API (S5, S6).

The official eBay *Marketplace Insights* API (real sold prices) is a gated,
manually-approved product. We avoid it and read the sold prices eBay already
exposes in its public search HTML with the ``LH_Sold=1&LH_Complete=1`` filters::

    https://www.ebay.com/sch/i.html?_nkw=iphone+13+pro&LH_Sold=1&LH_Complete=1

This is the project's **historical ground-truth resale signal** — actual
completed sales over roughly the last 90 days, each with a sale date. Paginating
with ``_pgn`` at ``_ipg=240`` yields hundreds per model and thousands overall.

Two robustness problems this scraper solves:

1. **Bot wall.** eBay fronts these pages with Akamai, which fingerprints the TLS
   (JA3) handshake and serves a tiny stub page to datacenter IPs even when the
   HTTP status is 200. We impersonate a real browser TLS with ``curl_cffi`` and
   **retry with fresh warmed sessions until a full results page comes back**
   (detected by page size + card count). Residential IPs almost never block;
   cloud IPs get intermittent windows, so the retry loop scavenges them.
2. **Layout drift.** eBay A/B-tests two card layouts (``li.s-item`` and the newer
   ``li.s-card`` / ``su-card-container``). Rather than depend on fragile inner
   class names we locate cards by either container and extract price / title /
   sold-date by **regex over the card text** — which survives both layouts.

Prices on ``ebay.com`` are USD; we convert to EUR with a configurable rate so the
labels are comparable to the Spanish Wallapop/Backmarket data. (This is US-market
sold data used as a temporal/price anchor — see the README caveat.)
"""

from __future__ import annotations

import re
import time
from typing import Any, Iterable

from bs4 import BeautifulSoup

from ..exceptions import EbayScraperError
from ..schema import Listing
from .base import BaseScraper

# Price: "$1,234.56" (USD) or "1.234,56 EUR" / "EUR 300,00".
_USD_RE = re.compile(r"\$\s?([\d,]+(?:\.\d{2})?)")
_EUR_RE = re.compile(r"(?:EUR|€)\s?([\d.]+,\d{2})|([\d.]+,\d{2})\s?(?:EUR|€)")
# Sold date — English ("Sold Mar 15, 2026") or Spanish ("Vendidos 17 jun 2026").
_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7,
    "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "ene": 1, "abr": 4, "ago": 8, "dic": 12,  # Spanish-only abbreviations
}
_DATE_EN = re.compile(r"sold\s+([a-z]{3})\s+(\d{1,2}),?\s+(\d{4})", re.IGNORECASE)
_DATE_ES = re.compile(r"vendidos?\s+(\d{1,2})\s+([a-z]{3,4})\.?\s+(\d{4})", re.IGNORECASE)


def _parse_price(text: str) -> tuple[float, str] | None:
    """Return (amount, currency) from an eBay price string, or None."""
    if not text:
        return None
    m = _USD_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(",", "")), "USD"
        except ValueError:
            pass
    m = _EUR_RE.search(text)
    if m:
        raw = m.group(1) or m.group(2)
        raw = raw.replace(".", "").replace(",", ".")
        try:
            return float(raw), "EUR"
        except ValueError:
            pass
    return None


def _parse_sold_date_ms(text: str) -> float | None:
    """Parse a sold-date caption (English or Spanish) → epoch milliseconds."""
    if not text:
        return None
    import datetime as _dt

    m = _DATE_EN.search(text)
    if m:
        mon = _MONTHS.get(m.group(1).lower())
        day, year = int(m.group(2)), int(m.group(3))
    else:
        m = _DATE_ES.search(text)
        if not m:
            return None
        day = int(m.group(1)); mon = _MONTHS.get(m.group(2)[:3].lower()); year = int(m.group(3))
    if not mon:
        return None
    try:
        return _dt.datetime(year, mon, day, tzinfo=_dt.timezone.utc).timestamp() * 1000.0
    except ValueError:
        return None


class EbayScraper(BaseScraper):
    """Scrapes eBay sold/completed iPhone listings as historical resale data."""

    source_name = "ebay"

    def __init__(self, cfg: dict[str, Any], user_agent: str,
                 impersonate: str = "safari17_0"):
        super().__init__(
            user_agent=user_agent,
            rate_limit_seconds=cfg.get("rate_limit_seconds", 1.5),
            max_retries=cfg.get("max_retries", 2),
            impersonate=impersonate,
            warm_url=None,  # we warm per-attempt inside _fetch_real
        )
        self.base_url = cfg["base_url"]
        self.max_pages = cfg.get("max_pages", 4)
        self.items_per_page = cfg.get("items_per_page", 240)
        self.fetch_retries = cfg.get("fetch_retries", 8)
        self.usd_to_eur = cfg.get("usd_to_eur", 0.92)
        self.home_url = cfg.get("home_url", "https://www.ebay.com/")

    # --- intermittency-resistant fetch -----------------------------------
    def _fetch_real(self, params: dict) -> list:
        """Fetch one results page, retrying with fresh warmed sessions until a
        full page (not an Akamai stub) is returned. Returns the card list."""
        from urllib.parse import urlencode

        url = f"{self.base_url}?{urlencode(params)}"
        for attempt in range(1, self.fetch_retries + 1):
            self._throttle()
            try:
                # fresh session each attempt — a flagged session stays flagged
                if self._impersonate:
                    from curl_cffi import requests as creq
                    sess = creq.Session(impersonate=self._impersonate,
                                        headers=dict(self.session.headers))
                else:
                    sess = self.session
                sess.get(self.home_url, timeout=self.timeout)  # warm cookies
                time.sleep(0.6)
                resp = sess.get(url, timeout=self.timeout)
                soup = BeautifulSoup(resp.text, "lxml")
                cards = soup.select("li.s-card, li.s-item")
                cards = [c for c in cards if "Shop on eBay" not in c.get_text()]
                if len(cards) > 5:
                    self.log.debug("page ok after %d attempt(s): %d cards", attempt, len(cards))
                    return cards
                self.log.debug("stub page (len=%d) attempt %d/%d",
                               len(resp.text), attempt, self.fetch_retries)
            except Exception as exc:  # noqa: BLE001
                self.log.debug("fetch error attempt %d: %s", attempt, exc)
            time.sleep(min(2.0 * attempt, 8.0))
        return []

    def _search(self, model: str, city: str, **kwargs: Any) -> Iterable[dict]:
        records: list[dict] = []
        for page in range(1, self.max_pages + 1):
            params = {"_nkw": model, "LH_Sold": 1, "LH_Complete": 1,
                      "_ipg": self.items_per_page, "_pgn": page}
            cards = self._fetch_real(params)
            if not cards:
                if page == 1:
                    raise EbayScraperError(f"blocked/empty for '{model}' (IP likely rate-limited)")
                break
            records.extend({"card": c} for c in cards)
        return records

    # --- parse (regex over card text → robust to layout) -----------------
    def _parse(self, record: dict, *, model: str, city: str) -> Listing | None:
        card = record["card"]
        text = card.get_text(" ", strip=True)

        priced = _parse_price(text)
        if priced is None:
            return None
        amount, cur = priced
        if cur == "USD":
            amount *= self.usd_to_eur

        title_el = card.select_one(
            ".s-card__title, .s-item__title, [role='heading'], .su-styled-text"
        )
        title = title_el.get_text(" ", strip=True) if title_el else ""
        title = re.sub(r"\b(Opens? in a new window.*|Se abre en una.*|New Listing)\b",
                       "", title, flags=re.IGNORECASE).strip()
        if not title or len(title) < 5:
            return None

        link_el = card.select_one("a[href*='/itm/']")
        url = link_el.get("href") if link_el else None
        listing_id = "ebay-unknown"
        if url:
            m = re.search(r"/itm/(\d{6,})", url)
            if m:
                listing_id = f"ebay-{m.group(1)}"
        img_el = card.select_one("img")

        return Listing(
            listing_id=listing_id,
            source=self.source_name,
            title=title,
            description="",
            price=round(amount, 2),
            currency="EUR",
            city=None,
            country_code="US",
            image_url=(img_el.get("src") or img_el.get("data-src")) if img_el else None,
            n_photos=1 if img_el else 0,
            created_at=_parse_sold_date_ms(text),  # real SALE date → temporal depth (S6)
            url=url,
            scraped_at=self._now_iso(),
            query_model=model,
            query_city=city,
        )

    def sweep(self, models: list[str]) -> list[Listing]:
        out: list[Listing] = []
        for model in models:
            cell = self.scrape(model, city="-")
            out.extend(cell)
            self.log.info("eBay %-22s → %d sold (running total %d)", model, len(cell), len(out))
        self.log.info("eBay sweep complete: %d sold listings", len(out))
        return out
