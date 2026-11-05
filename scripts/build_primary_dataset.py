#!/usr/bin/env python
"""Baixa e processa a janela primária de forma incremental."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from hawkes_crypto.data.clean import concatenate_event_chunks, iter_book_ticker_events
from hawkes_crypto.data.download_binance import (
    DownloadRecord,
    download_daily_archive,
    extract_archive,
)

TICK_SIZES = {"BTCUSDT": 0.1, "ETHUSDT": 0.01}


def days_between(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def download_one(
    market: str, symbol: str, product: str, day: date, root: Path
) -> DownloadRecord:
    return download_daily_archive(market, product, symbol, day, root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2024-01-15")
    parser.add_argument("--end", default="2024-01-28")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim"))
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    jobs = [
        ("um_futures", symbol, product, day, args.raw_root)
        for day in days_between(start, end)
        for symbol in ("BTCUSDT", "ETHUSDT")
        for product in ("bookTicker", "aggTrades")
    ]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        records = list(executor.map(lambda job: download_one(*job), jobs))
    manifest = args.raw_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item.path):
            handle.write(json.dumps(record.__dict__, sort_keys=True) + "\n")

    quality: list[dict[str, object]] = []
    for record in records:
        if record.product != "bookTicker":
            continue
        extracted = extract_archive(
            record,
            args.interim_root / record.market / record.symbol / record.date,
        )
        csv_path = next(path for path in extracted if path.suffix == ".csv")
        output = (
            args.processed_root
            / record.market
            / record.symbol
            / f"{record.date}.parquet"
        )
        chunks = list(
            iter_book_ticker_events(
                csv_path,
                TICK_SIZES[record.symbol],
                chunksize=args.chunksize,
            )
        )
        events = concatenate_event_chunks(chunks)
        output.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(output, index=False)
        quality.append(
            {
                "market": record.market,
                "symbol": record.symbol,
                "date": record.date,
                "events": len(events),
                "segments_1000": len(events) // 1000,
                "output": str(output),
            }
        )
        csv_path.unlink()
    quality_path = args.processed_root / "quality_primary.json"
    quality_path.write_text(
        json.dumps(sorted(quality, key=lambda item: str(item)), indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"archives": len(records), "book_ticker_days": len(quality)}, indent=2))


if __name__ == "__main__":
    main()
