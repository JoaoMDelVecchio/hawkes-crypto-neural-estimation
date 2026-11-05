#!/usr/bin/env python
"""Audita bookTicker local e gera a sequência de eventos do piloto."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hawkes_crypto.data.clean import (
    audit_quotes,
    quotes_to_events,
    read_book_ticker_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--market", required=True, choices=["um_futures"])
    parser.add_argument("--symbol", required=True, choices=["BTCUSDT", "ETHUSDT"])
    parser.add_argument("--tick-size", type=float, required=True)
    parser.add_argument("--frequency", default="100ms")
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Limita o piloto; arquivos diários podem ter dezenas de milhões de linhas",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    quotes = read_book_ticker_csv(args.input, nrows=args.max_rows)
    audit = audit_quotes(quotes).to_dict()
    events = quotes_to_events(
        quotes, tick_size=args.tick_size, frequency=args.frequency
    )
    summary = {
        "source": str(args.input),
        "market": args.market,
        "symbol": args.symbol,
        "frequency": args.frequency,
        "max_rows": args.max_rows,
        "quote_audit": audit,
        "events": len(events),
        "segments_1000": len(events) // 1000,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        events.to_parquet(args.output.with_suffix(".parquet"), index=False)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
