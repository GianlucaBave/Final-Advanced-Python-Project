"""Core data containers (S1 — OOP).

``Listing`` and ``Deal`` use ``__slots__`` to (a) make the attribute contract
explicit, (b) prevent typos creating phantom attributes, and (c) cut per-object
memory — relevant when thousands of listings are held in memory during a scan.

These are deliberately *thin* records. All heavy logic lives in the pipeline,
model and scorer; these objects just move structured data between layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional

# Canonical column order for the raw scrape parquet. Keeping it in one place
# means scrapers, validator and loader never disagree about the schema.
RAW_COLUMNS: tuple[str, ...] = (
    "listing_id",
    "source",
    "title",
    "description",
    "price",
    "currency",
    "city",
    "region",
    "country_code",
    "image_url",
    "n_photos",
    "seller_id",
    "category_id",
    "created_at",        # epoch ms or ISO — normalized downstream
    "shipping_available",
    "is_refurbished",
    "has_warranty_flag",
    "url",
    "scraped_at",        # ISO8601 string, set by the scraper
    "query_model",       # the model string used to find this listing
    "query_city",
)


@dataclass(slots=True)
class Listing:
    """A single second-hand listing from any source, pre-feature-extraction."""

    listing_id: str
    source: str                      # "wallapop" | "backmarket" | "ebay"
    title: str
    price: float
    currency: str = "EUR"
    description: str = ""
    city: Optional[str] = None
    region: Optional[str] = None
    country_code: str = "ES"
    image_url: Optional[str] = None
    n_photos: int = 0
    seller_id: Optional[str] = None
    category_id: Optional[int] = None
    created_at: Optional[Any] = None
    shipping_available: bool = False
    is_refurbished: bool = False
    has_warranty_flag: bool = False
    url: Optional[str] = None
    scraped_at: Optional[str] = None
    query_model: Optional[str] = None
    query_city: Optional[str] = None

    def as_row(self) -> dict[str, Any]:
        """Flatten to a dict ordered like :data:`RAW_COLUMNS` for DataFrame I/O."""
        d = {f.name: getattr(self, f.name) for f in fields(self)}
        return {col: d.get(col) for col in RAW_COLUMNS}


@dataclass(slots=True)
class Deal:
    """A scored listing — the output of the investment pipeline."""

    listing_id: str
    title: str
    asking_price: float
    predicted_fair_price: float
    price_low: float                 # 10th percentile
    price_high: float                # 90th percentile
    expected_margin: float
    confidence: float
    risk_score: float
    investment_score: float
    decision: str                    # STRONG BUY | BUY | HOLD | SKIP
    risk_flags: list[str] = field(default_factory=list)
    city: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}
