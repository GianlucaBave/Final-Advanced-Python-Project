"""Fetch a single Wallapop listing from its public URL.

Wallapop exposes two different identifiers for the same listing:
the URL slug (``iphone-13-tienda-garantia-1273335122``) and the API id
(``8z880moe81z3``). Only the slug is visible in the user-facing URL, so the
matcher resolves a listing by **slug equality** against the search endpoint —
the same API the rest of the scraper already uses.

Strategy:

1. Extract the slug from the URL.
2. Derive search keywords from the slug.
3. Call :class:`WallapopScraper` with those keywords across a small set of
   cities and return the first listing whose ``web_slug`` matches.

Searches are isolated per-city so a single rate-limit hiccup doesn't break the
whole resolution.
"""

from __future__ import annotations

import re
from typing import Iterable, Optional

from ..logging_setup import get_logger
from ..schema import Listing
from ..scrapers.wallapop import WallapopScraper

log = get_logger(__name__)

_ITEM_PATH_RE = re.compile(r"/item/([^/?#]+)")
_TRAILING_ID_RE = re.compile(r"-\d{6,}$")


# --- URL parsing ---------------------------------------------------------
def extract_slug(url: str) -> Optional[str]:
    """Pull the ``web_slug`` portion out of a Wallapop item URL."""
    if not url:
        return None
    m = _ITEM_PATH_RE.search(url.strip())
    if not m:
        return None
    return m.group(1).rstrip("/")


def slug_to_keywords(slug: str, max_words: int = 5) -> str:
    """Derive a short Wallapop search query from a listing slug."""
    base = _TRAILING_ID_RE.sub("", slug)
    words = [w for w in base.split("-") if w]
    return " ".join(words[:max_words])


# --- matching ------------------------------------------------------------
def _matches(listing: Listing, slug: str) -> bool:
    if not listing.url:
        return False
    listing_slug = extract_slug(listing.url)
    return listing_slug == slug


# --- public API ----------------------------------------------------------
def fetch_listing(
    url: str,
    scraper: WallapopScraper,
    cities: Optional[Iterable[str]] = None,
) -> Listing:
    """Resolve a Wallapop URL to a :class:`Listing` or raise ``ValueError``.

    Parameters
    ----------
    url
        Public Wallapop item URL (``https://es.wallapop.com/item/...``).
    scraper
        Configured :class:`WallapopScraper` whose ``cities`` map is used as the
        search anchor pool.
    cities
        Optional override for the city sweep. Defaults to the scraper's full
        configured list.
    """
    slug = extract_slug(url)
    if not slug:
        raise ValueError(
            "That doesn't look like a Wallapop item URL. "
            "Expected something like https://es.wallapop.com/item/iphone-..."
        )

    keywords = slug_to_keywords(slug) or "iPhone"
    city_pool = list(cities) if cities is not None else list(scraper.cities.keys())
    if not city_pool:
        raise ValueError("No cities configured to search from.")

    log.info("resolving slug=%r via search keywords=%r in %d cities",
             slug, keywords, len(city_pool))
    for city in city_pool:
        try:
            results = scraper.scrape(keywords, city)
        except Exception as exc:  # noqa: BLE001
            log.warning("search in %s failed: %s", city, exc)
            continue
        for listing in results:
            if _matches(listing, slug):
                log.info("matched listing %s in %s (€%.0f)",
                         listing.listing_id, city, listing.price)
                return listing

    raise ValueError(
        "Couldn't find this listing on Wallapop right now. "
        "It may have been removed, or it's outside the cities we search."
    )
