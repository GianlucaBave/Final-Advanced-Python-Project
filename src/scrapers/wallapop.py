"""Wallapop scraper — live, working, no auth (S5).

Wallapop's web app is a single-page React application that calls its own JSON
backend at ``api.wallapop.com/api/v3/search``. We consume that same endpoint
the browser does, which yields clean structured JSON — far more robust than
parsing the client-rendered HTML (BeautifulSoup would see an empty shell).

Verified working endpoint/shape (2026-06):

    GET https://api.wallapop.com/api/v3/search
        ?source=search_box&keywords=<model>&latitude=<lat>&longitude=<lon>
    headers: realistic UA + Accept: application/json + X-DeviceOS: 0
             + Origin/Referer es.wallapop.com
    body: { data: { section: { payload: { items: [ ... ] } } }, meta: {...} }

Each item exposes id, title, description (where battery %, condition and
accessories live as free text), price.amount, location.city/region,
created_at (epoch ms), shipping, has_warranty, is_refurbished and images[].
"""

from __future__ import annotations

from typing import Any, Iterable
from urllib.parse import quote

from ..exceptions import WallapopScraperError
from ..schema import Listing
from .base import BaseScraper


class WallapopScraper(BaseScraper):
    """Scrapes second-hand iPhone listings from Wallapop's JSON search API."""

    source_name = "wallapop"

    def __init__(self, cfg: dict[str, Any], cities: dict[str, dict]):
        wp = cfg
        super().__init__(
            user_agent=wp["user_agent"],
            rate_limit_seconds=wp.get("rate_limit_seconds", 1.0),
            max_retries=wp.get("max_retries", 3),
            backoff_base=wp.get("backoff_base", 2.0),
            extra_headers={
                "Accept": "application/json",
                "X-DeviceOS": "0",
                "Origin": "https://es.wallapop.com",
                "Referer": "https://es.wallapop.com/",
            },
        )
        self.base_url = wp["base_url"]
        self.category_id = wp.get("category_id")
        self.page_size = wp.get("page_size", 40)
        self.max_pages = wp.get("max_pages_per_query", 1)
        self.cities = cities

    # --- BaseScraper hooks ------------------------------------------------
    def _search(self, model: str, city: str, **kwargs: Any) -> Iterable[dict]:
        coords = self.cities.get(city)
        if coords is None:
            raise WallapopScraperError(f"unknown city '{city}' (no coordinates)")

        params = {
            "source": "search_box",
            "keywords": model,
            "latitude": coords["lat"],
            "longitude": coords["lon"],
            "order_by": "most_relevance",
        }
        if self.category_id:
            params["category_ids"] = self.category_id

        items: list[dict] = []
        next_page: str | None = None
        for page in range(self.max_pages):
            q = dict(params)
            if next_page:
                q["next_page"] = next_page
            resp = self._request(self.base_url, params=q)
            try:
                payload = resp.json()
            except ValueError as exc:
                raise WallapopScraperError(f"non-JSON response: {exc}") from exc

            page_items = (
                payload.get("data", {})
                .get("section", {})
                .get("payload", {})
                .get("items", [])
            )
            if not page_items:
                break
            items.extend(page_items)

            next_page = payload.get("meta", {}).get("next_page")
            if not next_page:
                break
        return items

    def _parse(self, record: dict, *, model: str, city: str) -> Listing | None:
        price_obj = record.get("price") or {}
        amount = price_obj.get("amount")
        if amount is None:
            return None

        images = record.get("images") or []
        image_url = None
        if images:
            urls = images[0].get("urls") or {}
            image_url = urls.get("big") or urls.get("medium") or urls.get("small")

        loc = record.get("location") or {}
        shipping = record.get("shipping") or {}
        web_slug = record.get("web_slug")
        url = f"https://es.wallapop.com/item/{web_slug}" if web_slug else None

        return Listing(
            listing_id=str(record.get("id")),
            source=self.source_name,
            title=record.get("title", ""),
            description=record.get("description", "") or "",
            price=float(amount),
            currency=price_obj.get("currency", "EUR"),
            city=loc.get("city") or city.title(),
            region=loc.get("region"),
            country_code="ES",
            image_url=image_url,
            n_photos=len(images),
            seller_id=record.get("user_id"),
            category_id=record.get("category_id"),
            created_at=record.get("created_at"),
            shipping_available=bool(shipping.get("item_is_shippable", False)),
            is_refurbished=bool(record.get("is_refurbished", False)),
            has_warranty_flag=bool(record.get("has_warranty", False)),
            url=url,
            scraped_at=self._now_iso(),
            query_model=model,
            query_city=city,
        )

    # --- convenience sweep ------------------------------------------------
    def sweep(self, models: list[str], cities: list[str]) -> list[Listing]:
        """Scrape the full models × cities grid. Errors per cell are isolated."""
        all_listings: list[Listing] = []
        failures = 0
        total = len(models) * len(cities)
        for model in models:
            for city in cities:
                cell = self.scrape(model, city)
                if not cell:
                    failures += 1
                all_listings.extend(cell)
        if total and failures / total > 0.10:
            import warnings

            warnings.warn(
                f"Wallapop sweep: {failures}/{total} queries returned nothing "
                "(>10%) — possible rate-limiting or API change.",
                UserWarning,
                stacklevel=2,
            )
        self.log.info(
            "Wallapop sweep complete: %d listings from %d queries (%d empty)",
            len(all_listings), total, failures,
        )
        return all_listings
