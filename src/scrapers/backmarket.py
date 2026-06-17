"""Backmarket scraper — professional refurbished anchors via JSON-LD (S5).

Backmarket (a refurbished-electronics marketplace) gives us *firm* professional
prices per SKU/condition — clean anchors that tell the model what a graded
device is worth. Rather than scraping fragile CSS classes we read the
``<script type="application/ld+json">`` Product/Offer blocks that Backmarket
embeds for SEO; these carry name, price, currency and condition.

Same bot-protection caveat as eBay applies: Backmarket fronts pages with
PerimeterX and returns 403 to datacenter IPs. Implemented and wired; degrades
gracefully when blocked.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable
from urllib.parse import quote_plus

from bs4 import BeautifulSoup

from ..exceptions import BackmarketScraperError
from ..schema import Listing
from .base import BaseScraper


def _iter_products(node: Any) -> Iterable[dict]:
    """Recursively yield JSON-LD nodes whose @type is Product."""
    if isinstance(node, dict):
        t = node.get("@type")
        if t == "Product" or (isinstance(t, list) and "Product" in t):
            yield node
        for v in node.values():
            yield from _iter_products(v)
    elif isinstance(node, list):
        for v in node:
            yield from _iter_products(v)


class BackmarketScraper(BaseScraper):
    """Scrapes Backmarket Spain refurbished iPhone reference prices."""

    source_name = "backmarket"

    def __init__(self, cfg: dict[str, Any], user_agent: str,
                 impersonate: str = "chrome124"):
        super().__init__(
            user_agent=user_agent,
            rate_limit_seconds=cfg.get("rate_limit_seconds", 1.5),
            max_retries=cfg.get("max_retries", 2),
            impersonate=impersonate,
            warm_url="https://www.backmarket.es/",
        )
        self.base_url = cfg["base_url"]

    def _search(self, model: str, city: str, **kwargs: Any) -> Iterable[dict]:
        url = f"{self.base_url}/es-es/search?q={quote_plus(model)}"
        resp = self._request(url)
        soup = BeautifulSoup(resp.text, "lxml")
        blocks = soup.find_all("script", attrs={"type": "application/ld+json"})
        if not blocks:
            raise BackmarketScraperError("no JSON-LD blocks (layout change or block)")
        products: list[dict] = []
        for block in blocks:
            try:
                data = json.loads(block.string or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            products.extend(_iter_products(data))
        return products

    def _parse(self, record: dict, *, model: str, city: str) -> Listing | None:
        offers = record.get("offers") or {}
        if isinstance(offers, list):
            offers = offers[0] if offers else {}
        price = offers.get("price") or offers.get("lowPrice")
        if price is None:
            return None
        try:
            price = float(price)
        except (TypeError, ValueError):
            return None

        sku = record.get("sku") or record.get("mpn") or record.get("name", "")
        return Listing(
            listing_id=f"backmarket-{re.sub(r'[^A-Za-z0-9]+', '-', str(sku))[:48]}",
            source=self.source_name,
            title=record.get("name", model),
            description=record.get("description", "") or "",
            price=price,
            currency=offers.get("priceCurrency", "EUR"),
            city=None,
            country_code="ES",
            image_url=(record.get("image") if isinstance(record.get("image"), str) else None),
            n_photos=1,
            is_refurbished=True,        # Backmarket is refurbished by definition
            has_warranty_flag=True,     # Backmarket includes warranty
            url=record.get("url") or record.get("@id"),
            scraped_at=self._now_iso(),
            query_model=model,
            query_city=city,
        )

    def sweep(self, models: list[str]) -> list[Listing]:
        out: list[Listing] = []
        for model in models:
            out.extend(self.scrape(model, city="-"))
        self.log.info("Backmarket sweep complete: %d reference prices", len(out))
        return out
