"""Command-line interface — ``python -m src.cli <command>`` (S9.6).

Implemented with the stdlib ``argparse`` (no extra dependency). Commands:

    scrape    run the live Wallapop sweep → data/raw/*.parquet
    train     train all models from the latest scrape → artifacts/
    scan      find deals for a model/city on a fresh live scrape
    evaluate  print the latest training report

Example::

    python -m src.cli scrape
    python -m src.cli train --tune
    python -m src.cli scan --city madrid --model "iPhone 13 Pro" --max-price 700
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .config import load_config
from .data import listings_to_frame, save_parquet, timestamped_path, validate_listings
from .logging_setup import configure_logging, get_logger
from .reporting import default_title, render_deals

log = get_logger("cli")


# --- commands ------------------------------------------------------------
def cmd_scrape(args, cfg) -> int:
    from .scrapers import BackmarketScraper, EbayScraper, WallapopScraper

    models = [args.model] if args.model else cfg.models
    cities = [args.city] if args.city else list(cfg.cities.keys())
    ua = cfg.scraping["wallapop"]["user_agent"]
    sources = args.sources.split(",") if args.sources else ["wallapop"]
    written = []

    if "wallapop" in sources:
        wp = WallapopScraper(cfg.scraping["wallapop"], cfg.cities)
        df = validate_listings(listings_to_frame(wp.sweep(models, cities)), source="wallapop")
        df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)
        p = save_parquet(df, timestamped_path(cfg.paths.raw_dir, "wallapop_iphones"))
        written.append((len(df), p))

    if "ebay" in sources:
        eb = EbayScraper(cfg.scraping["ebay"], ua)
        df = validate_listings(listings_to_frame(eb.sweep(models)), source="ebay")
        df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)
        if len(df):
            p = save_parquet(df, timestamped_path(cfg.paths.raw_dir, "ebay_sold"))
            written.append((len(df), p))
        else:
            print("eBay returned no rows (bot-blocked from this IP — try a residential network).")

    if "backmarket" in sources:
        bm = BackmarketScraper(cfg.scraping["backmarket"], ua)
        df = validate_listings(listings_to_frame(bm.sweep(models)), source="backmarket")
        df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)
        if len(df):
            p = save_parquet(df, timestamped_path(cfg.paths.raw_dir, "backmarket_reference"))
            written.append((len(df), p))
        else:
            print("Backmarket returned no rows (bot-blocked from this IP).")

    for n, p in written:
        print(f"Scraped {n} unique listings → {p}")
    return 0


def cmd_train(args, cfg) -> int:
    from .training import train_all

    report = train_all(cfg, tune=args.tune)
    m = report["models"]
    print("\n=== Training complete ===")
    for name, res in m.items():
        t = res["test"]
        print(f"  {name:16s} test  MAE €{t['mae']:.1f}  MAPE {t['mape']:.1f}%  R² {t['r2']:.3f}")
    q = report.get("quantile", {})
    if q:
        print(f"  quantile interval coverage: {q['empirical_coverage']:.1%} "
              f"(nominal {q['nominal']:.0%})")
    print(f"  artifacts → {cfg.paths.artifacts_dir}")
    return 0


def cmd_scan(args, cfg) -> int:
    from .features.parser import is_accessory_listing
    from .investment import DealDetector
    from .scrapers import WallapopScraper

    detector = DealDetector.from_artifacts(cfg)
    wp = WallapopScraper(cfg.scraping["wallapop"], cfg.cities)

    model = args.model or "iPhone"
    cities = [args.city] if args.city else ["madrid"]
    listings = []
    for city in cities:
        listings += wp.scrape(model, city)
    df = validate_listings(listings_to_frame(listings), source="wallapop")
    # Apply the same accessory/parts hygiene used at training time so we never
    # recommend a case or a spare screen as a "deal".
    df = df[~df["title"].fillna("").map(is_accessory_listing)]
    if args.max_price:
        df = df[df["price"] <= args.max_price]
    df = df.drop_duplicates(subset=["listing_id"]).reset_index(drop=True)

    if df.empty:
        print("No live listings matched the query.")
        return 0

    deals = detector.detect(df)
    title = default_title(args.model, args.city)
    print(render_deals(deals, title=title, top=args.top))

    # persist JSON report (timestamped for reproducibility)
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = cfg.paths.reports_dir / f"deals_{stamp}.json"
    detector.save_report(
        deals, report_path, top=args.top,
        meta={"model": args.model, "city": args.city, "max_price": args.max_price,
              "n_listings_scanned": int(len(df))},
    )
    print(f"\nFull report ({len(deals)} scored) → {report_path}")
    return 0


def cmd_evaluate(args, cfg) -> int:
    path = cfg.paths.artifacts_dir / "training_report.json"
    if not path.exists():
        print("No training report found. Run `train` first.")
        return 1
    report = json.loads(path.read_text(encoding="utf-8"))
    print(json.dumps(report, indent=2)[:4000])
    return 0


# --- argument parser -----------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="iphone-deal-finder",
                                description="Find under-priced second-hand iPhones on Wallapop.")
    p.add_argument("--config", default=None, help="path to settings.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("scrape", help="run the live scrapers")
    s.add_argument("--model", default=None, help="restrict to one model")
    s.add_argument("--city", default=None, help="restrict to one city")
    s.add_argument("--sources", default="wallapop",
                   help="comma-separated: wallapop,ebay,backmarket")
    s.set_defaults(func=cmd_scrape)

    t = sub.add_parser("train", help="train all models from the latest scrape")
    t.add_argument("--tune", action="store_true", help="run GridSearchCV tuning")
    t.set_defaults(func=cmd_train)

    sc = sub.add_parser("scan", help="find deals on a fresh live scrape")
    sc.add_argument("--model", default=None, help='e.g. "iPhone 13 Pro"')
    sc.add_argument("--city", default="madrid")
    sc.add_argument("--max-price", type=float, default=None)
    sc.add_argument("--top", type=int, default=10)
    sc.set_defaults(func=cmd_scan)

    e = sub.add_parser("evaluate", help="print the latest training report")
    e.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    configure_logging(
        console_level=cfg.logging.get("console_level", "INFO"),
        file_level=cfg.logging.get("file_level", "DEBUG"),
    )
    try:
        return args.func(args, cfg)
    except Exception as exc:  # noqa: BLE001
        log.exception("command failed: %s", exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
