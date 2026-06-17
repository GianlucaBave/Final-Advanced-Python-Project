"""Multi-source merge + dedup (S2 — vectorized).

Combines validated batches from each source into a single training table:

* concatenates the per-source frames,
* deduplicates within a source on ``listing_id`` (a re-scrape of the same item),
* assigns a ``sample_weight`` per row from the configured source quality
  weights — used by the model so high-quality labels (eBay sold, Backmarket
  firm prices) count more than noisy Wallapop asking prices.

In the Wallapop-only v1 the weights collapse to a constant, which is a no-op.
"""

from __future__ import annotations

import pandas as pd

from ..logging_setup import get_logger

log = get_logger(__name__)


def merge_sources(
    frames: dict[str, pd.DataFrame],
    source_weights: dict[str, float],
) -> pd.DataFrame:
    """Concatenate per-source frames, dedup, and attach ``sample_weight``.

    Parameters
    ----------
    frames:
        ``{source_name: validated_frame}``. Empty/absent sources are skipped.
    source_weights:
        ``{source_name: weight}`` relative label-quality weights.
    """
    present = {k: v for k, v in frames.items() if v is not None and len(v)}
    if not present:
        return pd.DataFrame()

    parts = []
    for source, df in present.items():
        df = df.copy()
        # Dedup repeated scrapes of the same listing within the source.
        before = len(df)
        df = df.drop_duplicates(subset=["listing_id"], keep="last")
        if before - len(df):
            log.info("[%s] dropped %d duplicate listing_ids", source, before - len(df))
        df["source"] = source
        df["sample_weight"] = float(source_weights.get(source, 1.0))
        parts.append(df)

    merged = pd.concat(parts, ignore_index=True, sort=False)

    # Normalize weights so the mean weight is 1.0 (keeps XGBoost's effective
    # learning rate comparable regardless of the absolute weight scale).
    if merged["sample_weight"].mean() > 0:
        merged["sample_weight"] = merged["sample_weight"] / merged["sample_weight"].mean()

    log.info(
        "merged %d rows from %d sources: %s",
        len(merged), len(present), {k: len(v) for k, v in present.items()},
    )
    return merged
