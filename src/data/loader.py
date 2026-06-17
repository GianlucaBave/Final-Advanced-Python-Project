"""Parquet I/O (S2 — performance).

Parquet is columnar and compressed: smaller on disk and far faster to read than
CSV, and it preserves dtypes (no re-parsing dates/numbers on load). Every raw
scrape, interim and processed table goes through here so the storage format is
consistent and timestamped for reproducibility.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..logging_setup import get_logger
from ..schema import RAW_COLUMNS, Listing

log = get_logger(__name__)


def listings_to_frame(listings: list[Listing]) -> pd.DataFrame:
    """Vectorized build of a DataFrame from Listing objects (S2)."""
    rows = [ls.as_row() for ls in listings]
    df = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    return df


def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    log.info("wrote %d rows → %s", len(df), path)
    return path


def load_parquet(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path, engine="pyarrow")
    log.debug("read %d rows ← %s", len(df), path)
    return df


def timestamped_path(directory: str | Path, prefix: str, stamp: str | None = None) -> Path:
    """Build ``<directory>/<prefix>_YYYYMMDD.parquet`` for reproducible runs."""
    stamp = stamp or datetime.now().strftime("%Y%m%d")
    return Path(directory) / f"{prefix}_{stamp}.parquet"


def latest_parquet(directory: str | Path, prefix: str) -> Path | None:
    """Return the most recent ``<prefix>_*.parquet`` in *directory*, if any."""
    directory = Path(directory)
    if not directory.exists():
        return None
    matches = sorted(directory.glob(f"{prefix}_*.parquet"))
    return matches[-1] if matches else None
