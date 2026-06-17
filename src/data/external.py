"""Ingest external eBay CSV datasets into the project's raw schema.

The ``archive-5`` folder holds eBay iPhone price records exported to CSV
(columns ``Title``, ``Price``) — captured in **USD** on the **US** market around
**Nov 2023**. Prices may be single values or ranges ("$549 to $579"); we take the
midpoint, convert USD→EUR for comparability with the Spanish data, and stamp them
with the dataset's capture date so the out-of-time split treats them as the
oldest training data.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ..logging_setup import get_logger
from ..schema import RAW_COLUMNS

log = get_logger(__name__)

_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)")
# archive-5 was exported 2023-11-16; use as created_at for all its rows.
_CAPTURE_MS = datetime(2023, 11, 16, tzinfo=timezone.utc).timestamp() * 1000.0


def _parse_price_usd(text: str) -> float | None:
    """Parse a price cell → USD float. Ranges ('$549 to $579') → midpoint."""
    if not isinstance(text, str):
        return None
    vals = [float(m.replace(",", "")) for m in _MONEY.findall(text)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def ingest_ebay_csv_dir(
    csv_dir: str | Path,
    usd_to_eur: float = 0.92,
    source: str = "ebay_archive",
) -> pd.DataFrame:
    """Read every ``*.csv`` in *csv_dir* into a raw-schema DataFrame (EUR)."""
    csv_dir = Path(csv_dir)
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs in {csv_dir}")

    rows: list[dict] = []
    for f in files:
        df = pd.read_csv(f)
        cols = {c.lower(): c for c in df.columns}
        tcol, pcol = cols.get("title"), cols.get("price")
        if not tcol or not pcol:
            log.warning("skipping %s — missing Title/Price columns", f.name)
            continue
        for i, r in df.iterrows():
            title = str(r[tcol]).replace("NEW LISTING", "").strip()
            usd = _parse_price_usd(str(r[pcol]))
            if usd is None or not title:
                continue
            rows.append({
                "listing_id": f"{source}-{f.stem}-{i}",
                "source": source,
                "title": title,
                "description": "",
                "price": round(usd * usd_to_eur, 2),
                "currency": "EUR",
                "city": None,
                "region": None,
                "country_code": "US",
                "image_url": None,
                "n_photos": 1,
                "seller_id": None,
                "category_id": None,
                "created_at": _CAPTURE_MS,
                "shipping_available": False,
                "is_refurbished": False,
                "has_warranty_flag": False,
                "url": None,
                "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "query_model": f.stem.replace("ebay_", "").replace("_", " "),
                "query_city": None,
            })
    out = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    log.info("ingested %d rows from %d CSVs in %s", len(out), len(files), csv_dir)
    return out


_LOT_RE = re.compile(r"lot of|\bbundle\b|x10|pack of|\blote\b", re.IGNORECASE)


def ingest_ecommerce_csv(
    path: str | Path,
    usd_to_eur: float = 0.92,
    source: str = "ecommerce_us",
) -> pd.DataFrame:
    """Ingest the structured USA iPhone resale CSV (2026) into the raw schema.

    Richer than archive-5: carries ``condition``, ``storage_gb_numeric``,
    ``itemLocation`` and ``lastUpdated``. Prices are USD → converted to EUR; bulk
    "Lot of N" records are dropped; ``lastUpdated`` becomes ``created_at`` for the
    out-of-time split.
    """
    path = Path(path)
    df = pd.read_csv(path)
    price = pd.to_numeric(df.get("price"), errors="coerce")
    lots = df.get("title", "").astype(str).str.contains(_LOT_RE, na=False)
    updated = pd.to_datetime(df.get("lastUpdated"), errors="coerce")
    default_ms = datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp() * 1000.0

    rows: list[dict] = []
    for i, r in df.iterrows():
        if pd.isna(price.iloc[i]) or lots.iloc[i]:
            continue
        title = str(r.get("title", "")).strip()
        if not title:
            continue
        ts = updated.iloc[i]
        created_ms = ts.timestamp() * 1000.0 if pd.notna(ts) else default_ms
        cond = r.get("condition")
        rows.append({
            "listing_id": f"{source}-{i}",
            "source": source,
            "title": title,
            # surface the structured condition as text so the parser can read it
            "description": f"condition: {cond}" if isinstance(cond, str) else "",
            "price": round(float(price.iloc[i]) * usd_to_eur, 2),
            "currency": "EUR",
            "city": (str(r.get("itemLocation")) if pd.notna(r.get("itemLocation")) else None),
            "region": (str(r.get("us_state")) if pd.notna(r.get("us_state")) else None),
            "country_code": "US",
            "image_url": None,
            "n_photos": 1,
            "seller_id": (str(r.get("seller")) if pd.notna(r.get("seller")) else None),
            "category_id": None,
            "created_at": created_ms,
            "shipping_available": False,
            "is_refurbished": False,
            "has_warranty_flag": False,
            "url": None,
            "scraped_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "query_model": (str(r.get("model_family")) if pd.notna(r.get("model_family")) else None),
            "query_city": None,
        })
    out = pd.DataFrame(rows, columns=list(RAW_COLUMNS))
    log.info("ingested %d rows from %s", len(out), path.name)
    return out
