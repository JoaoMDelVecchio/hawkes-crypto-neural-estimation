#!/usr/bin/env python
"""Converte um bookTicker diário em Parquet de eventos usando chunks."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hawkes_crypto.data.clean import iter_book_ticker_events


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tick-size", type=float, required=True)
    parser.add_argument("--frequency", default="100ms")
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    args = parser.parse_args()

    chunks = list(
        iter_book_ticker_events(
            args.input,
            args.tick_size,
            frequency=args.frequency,
            chunksize=args.chunksize,
        )
    )
    events = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(args.output, index=False)
    print(f"{len(events)} eventos gravados em {args.output}")


if __name__ == "__main__":
    main()
