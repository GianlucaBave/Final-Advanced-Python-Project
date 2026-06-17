"""Scraper package — one class per source, all sharing :class:`BaseScraper`."""

from .base import BaseScraper
from .backmarket import BackmarketScraper
from .ebay import EbayScraper
from .wallapop import WallapopScraper

__all__ = [
    "BaseScraper",
    "WallapopScraper",
    "BackmarketScraper",
    "EbayScraper",
]
