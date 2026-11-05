#!/usr/bin/env python
"""Baixa arquivos diários e registra um manifesto verificável."""

import argparse
from datetime import date, timedelta
from pathlib import Path

from hawkes_crypto.data.download_binance import (
    append_manifest,
    download_daily_archive,
    extract_archive,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True, choices=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--product", required=True, choices=["bookTicker", "aggTrades"])
    parser.add_argument("--market", required=True, choices=["spot", "um_futures"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    parser.add_argument("--extract-to", type=Path)
    args = parser.parse_args()

    current = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if end < current:
        parser.error("--end deve ser igual ou posterior a --start")

    while current <= end:
        record = download_daily_archive(
            args.market, args.product, args.symbol, current, args.output
        )
        append_manifest(record, args.output / "manifest.jsonl")
        print(f"{current}: {record.bytes} bytes sha256={record.sha256}")
        if args.extract_to:
            extracted = extract_archive(
                record,
                args.extract_to
                / args.market
                / args.symbol
                / current.isoformat(),
            )
            for path in extracted:
                print(f"extraído: {path}")
        current += timedelta(days=1)
