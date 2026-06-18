"""Minimal Flask web UI for the iPhone Deal-Finder.

Run with::

    python -m src.webapp.app

Then open http://127.0.0.1:5000 and paste a Wallapop iPhone URL.

This module loads the trained artifacts once at startup and reuses them for
every request — the same :class:`DealDetector` the CLI ``scan`` command uses,
so train/serve parity holds.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

from ..config import load_config
from ..data import listings_to_frame
from ..features.parser import is_accessory_listing
from ..investment import DealDetector
from ..logging_setup import configure_logging, get_logger
from ..scrapers.wallapop import WallapopScraper
from .single_fetcher import fetch_listing

log = get_logger(__name__)


# --- app factory ---------------------------------------------------------
def create_app() -> Flask:
    cfg = load_config()
    configure_logging(
        console_level=cfg.logging.get("console_level", "INFO"),
        file_level=cfg.logging.get("file_level", "DEBUG"),
    )

    log.info("loading artifacts for webapp …")
    detector = DealDetector.from_artifacts(cfg)
    scraper = WallapopScraper(cfg.scraping["wallapop"], cfg.cities)

    template_dir = Path(__file__).resolve().parent / "templates"
    app = Flask(
        __name__,
        template_folder=str(template_dir),
    )

    @app.get("/")
    def index():
        return render_template("index.html", result=None, error=None, url="")

    @app.post("/analyze")
    def analyze():
        url = (request.form.get("url") or "").strip()
        if not url:
            return render_template(
                "index.html",
                result=None,
                error="Please paste a Wallapop iPhone listing URL.",
                url="",
            )
        try:
            listing = fetch_listing(url, scraper)
        except ValueError as exc:
            return render_template("index.html", result=None, error=str(exc), url=url)
        except Exception as exc:  # noqa: BLE001
            log.exception("fetch failed: %s", exc)
            return render_template(
                "index.html",
                result=None,
                error="Something went wrong while fetching that listing. Try again in a moment.",
                url=url,
            )

        # Accessory / parts hygiene: same filter the training pipeline applies.
        if is_accessory_listing(listing.title or ""):
            return render_template(
                "index.html",
                result=None,
                error="This looks like an accessory or spare part rather than an iPhone — the model only scores complete devices.",
                url=url,
            )

        df = listings_to_frame([listing])
        deals = detector.detect(df)
        if not deals:
            return render_template(
                "index.html",
                result=None,
                error="Couldn't score this listing — the parser found no recognisable iPhone in the title.",
                url=url,
            )

        deal = deals[0]
        result = {
            "title": deal.title,
            "image_url": listing.image_url,
            "city": deal.city or "—",
            "asking_price": f"€{deal.asking_price:,.0f}",
            "fair_price": f"€{deal.predicted_fair_price:,.0f}",
            "low": f"€{deal.price_low:,.0f}",
            "high": f"€{deal.price_high:,.0f}",
            "margin_pct": f"{deal.expected_margin * 100:+.1f}%",
            "margin_value": deal.expected_margin,
            "confidence": f"{deal.confidence:.2f}",
            "risk": f"{deal.risk_score:.2f}",
            "score": f"{deal.investment_score:.2f}",
            "score_value": deal.investment_score,
            "decision": deal.decision,
            "decision_class": _decision_class(deal.decision),
            "risk_flags": deal.risk_flags or [],
            "url": deal.url or url,
        }
        return render_template("index.html", result=result, error=None, url=url)

    return app


def _decision_class(decision: str) -> str:
    return {
        "STRONG BUY": "strong-buy",
        "BUY": "buy",
        "HOLD": "hold",
        "SKIP": "skip",
    }.get(decision.upper(), "hold")


def main() -> int:
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
