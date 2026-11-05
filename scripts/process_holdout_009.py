#!/usr/bin/env python
"""Processa a semana nova do ADR 009 (5--11/fev/2024) a partir do manifesto já baixado.

Os arquivos ``bookTicker`` desse período vêm com as linhas intercaladas fora da ordem do
tempo. Antes do mesmo processamento do período primário (``iter_book_ticker_events``, grade
de 100 ms), cada arquivo é ordenado por ``transaction_time`` e ``update_id``; o script
confere que as duas chaves concordam depois da ordenação e recusa o arquivo se não
concordarem.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_primary_dataset import TICK_SIZES  # noqa: E402

from hawkes_crypto.data.clean import (  # noqa: E402
    concatenate_event_chunks,
    iter_book_ticker_events,
)
from hawkes_crypto.data.download_binance import DownloadRecord, extract_archive  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw_holdout_009"))
    parser.add_argument("--interim-root", type=Path, default=Path("data/interim_holdout_009"))
    parser.add_argument("--processed-root", type=Path,
                        default=Path("data/processed_holdout_009"))
    args = parser.parse_args()
    records = [DownloadRecord(**json.loads(line))
               for line in (args.raw_root / "manifest.jsonl").read_text().splitlines()]
    quality = []
    for record in records:
        if record.product != "bookTicker":
            continue
        extracted = extract_archive(
            record, args.interim_root / record.market / record.symbol / record.date)
        csv_path = next(path for path in extracted if path.suffix == ".csv")
        frame = pd.read_csv(csv_path)
        was_sorted = bool(frame["transaction_time"].is_monotonic_increasing)
        frame = frame.sort_values(["transaction_time", "update_id"], kind="stable")
        if not frame["update_id"].is_monotonic_increasing:
            raise SystemExit(f"Ordem ambígua em {csv_path}: tempo e update_id discordam")
        frame.to_csv(csv_path, index=False)
        del frame
        events = concatenate_event_chunks(
            list(iter_book_ticker_events(csv_path, TICK_SIZES[record.symbol])))
        output = (args.processed_root / record.market / record.symbol
                  / f"{record.date}.parquet")
        output.parent.mkdir(parents=True, exist_ok=True)
        events.to_parquet(output, index=False)
        quality.append({"symbol": record.symbol, "date": record.date,
                        "file_was_time_sorted": was_sorted, "events": len(events),
                        "segments_1000": len(events) // 1000})
        csv_path.unlink()
        print(json.dumps(quality[-1]), flush=True)
    (args.processed_root / "quality_holdout_009.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
