"""Data validation (S4 — soft warnings vs hard errors).

The validator enforces *invariants* on a scraped batch. By design it is
forgiving: rows that violate a range/format rule are dropped with a ``WARNING``
log, never raising — one bad listing must not abort a 3,000-row scrape. It only
raises :class:`DataValidationError` for a *structural* problem (missing required
columns) where continuing would be meaningless.
"""

from __future__ import annotations

import warnings

import pandas as pd

from ..exceptions import DataValidationError
from ..logging_setup import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("listing_id", "source", "title", "price")
MAX_PRICE = 5000.0
MIN_PRICE = 0.0
MIN_TITLE_LEN = 5


def validate_listings(df: pd.DataFrame, *, source: str = "?") -> pd.DataFrame:
    """Return a cleaned copy of *df* with invalid rows dropped.

    Hard-fails only when required columns are absent. Everything else is a
    soft drop + warning so partial data still flows through.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"[{source}] missing required columns: {missing}")

    n0 = len(df)
    if n0 == 0:
        warnings.warn(f"[{source}] validation received an empty frame", UserWarning, stacklevel=2)
        return df.copy()

    df = df.copy()
    # Build a boolean keep-mask; report each failed rule.
    price = pd.to_numeric(df["price"], errors="coerce")
    title_len = df["title"].fillna("").astype(str).str.len()

    rules = {
        "price_not_numeric": price.isna(),
        "price_out_of_range": (price <= MIN_PRICE) | (price > MAX_PRICE),
        "title_too_short": title_len < MIN_TITLE_LEN,
        "missing_id": df["listing_id"].isna(),
    }
    bad = pd.Series(False, index=df.index)
    for name, mask in rules.items():
        n = int(mask.sum())
        if n:
            log.warning("[%s] dropping %d rows failing rule '%s'", source, n, name)
        bad = bad | mask.fillna(True)

    # Currency check is a warning only (we keep EUR rows, drop non-EUR).
    if "currency" in df.columns:
        non_eur = df["currency"].fillna("EUR").astype(str).str.upper() != "EUR"
        if int(non_eur.sum()):
            log.warning("[%s] dropping %d non-EUR rows", source, int(non_eur.sum()))
            bad = bad | non_eur

    clean = df.loc[~bad].copy()
    clean["price"] = pd.to_numeric(clean["price"], errors="coerce")
    dropped = n0 - len(clean)
    if dropped and dropped / n0 > 0.25:
        warnings.warn(
            f"[{source}] validation dropped {dropped}/{n0} rows (>25%) — "
            "check the scraper output quality.",
            UserWarning,
            stacklevel=2,
        )
    log.info("[%s] validated: kept %d / %d rows", source, len(clean), n0)
    return clean.reset_index(drop=True)
