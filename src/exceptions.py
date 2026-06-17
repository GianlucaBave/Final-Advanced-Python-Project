"""Custom exception hierarchy for iphone-deal-finder.

Every error is raised at the boundary where it occurs so that failures are
localized and can be caught at the right granularity. A caller that wants to
trap *any* project error catches :class:`DealFinderError`; a caller that only
cares about scraping catches :class:`ScraperError`; and so on.

    DealFinderError
    ├── ScraperError
    │   ├── WallapopScraperError
    │   ├── BackmarketScraperError
    │   └── EbayScraperError
    ├── DataValidationError
    ├── FeatureExtractionError
    ├── ModelTrainingError
    └── ModelInferenceError
"""

from __future__ import annotations


class DealFinderError(Exception):
    """Base class for every error raised by this package."""


# --- scraping (S5) -------------------------------------------------------
class ScraperError(DealFinderError):
    """Base class for any scraper failure."""


class WallapopScraperError(ScraperError):
    """Raised when the Wallapop search endpoint cannot be queried."""


class BackmarketScraperError(ScraperError):
    """Raised when Backmarket pages cannot be retrieved or parsed."""


class EbayScraperError(ScraperError):
    """Raised when eBay sold-listing pages cannot be retrieved or parsed."""


# --- data / features / model --------------------------------------------
class DataValidationError(DealFinderError):
    """Raised when a dataset fails a hard schema/range invariant."""


class FeatureExtractionError(DealFinderError):
    """Raised when feature construction fails irrecoverably."""


class ModelTrainingError(DealFinderError):
    """Raised when model training cannot complete (e.g. empty corpus)."""


class ModelInferenceError(DealFinderError):
    """Raised when a fitted artifact cannot score an input row."""
