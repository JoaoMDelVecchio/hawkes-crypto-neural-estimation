from datetime import date

import pandas as pd
import pytest

from hawkes_crypto.data.clean import (
    audit_quotes,
    concatenate_event_chunks,
    iter_book_ticker_events,
    quotes_to_events,
)
from hawkes_crypto.data.download_binance import daily_archive_url


def test_archive_urls_reflect_official_market_coverage() -> None:
    url = daily_archive_url(
        "um_futures", "bookTicker", "btcusdt", date(2024, 1, 15)
    )
    assert url.endswith(
        "/futures/um/daily/bookTicker/BTCUSDT/"
        "BTCUSDT-bookTicker-2024-01-15.zip"
    )
    with pytest.raises(ValueError, match="não está disponível"):
        daily_archive_url("spot", "bookTicker", "BTCUSDT", date(2024, 1, 15))


def test_quote_audit_and_event_transformation() -> None:
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00.000Z",
                    "2024-01-01T00:00:00.100Z",
                    "2024-01-01T00:00:00.200Z",
                    "2024-01-01T00:00:00.300Z",
                ],
                utc=True,
            ),
            "bid_price": [99.0, 100.0, 100.0, 99.0],
            "bid_qty": [1.0, 1.0, 1.0, 1.0],
            "ask_price": [101.0, 102.0, 102.0, 101.0],
            "ask_qty": [1.0, 1.0, 1.0, 1.0],
        }
    )
    audit = audit_quotes(quotes)
    assert audit.rows == 4
    assert audit.crossed_quotes == 0
    assert audit.median_update_ms == pytest.approx(100.0)

    events = quotes_to_events(quotes, tick_size=1.0)
    assert events["event_type"].tolist() == [1, 2]
    assert events["inter_arrival"].tolist() == pytest.approx([0.1, 0.2])
    assert events["jump_size_ticks"].tolist() == pytest.approx([1.0, 1.0])


def test_crossed_quotes_are_rejected() -> None:
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
            "bid_price": [102.0],
            "bid_qty": [1.0],
            "ask_price": [101.0],
            "ask_qty": [1.0],
        }
    )
    with pytest.raises(ValueError, match="Cotações inválidas"):
        quotes_to_events(quotes, tick_size=0.1)


def test_chunked_processing_preserves_bucket_boundaries(tmp_path) -> None:
    path = tmp_path / "book_ticker.csv"
    pd.DataFrame(
        {
            "update_id": range(6),
            "best_bid_price": [99, 99, 100, 100, 99, 99],
            "best_bid_qty": [1] * 6,
            "best_ask_price": [101, 101, 102, 102, 101, 101],
            "best_ask_qty": [1] * 6,
            "transaction_time": [
                1704067200000,
                1704067200050,
                1704067200110,
                1704067200150,
                1704067200250,
                1704067200350,
            ],
            "event_time": [
                1704067200000,
                1704067200050,
                1704067200110,
                1704067200150,
                1704067200250,
                1704067200350,
            ],
        }
    ).to_csv(path, index=False)

    events = pd.concat(
        list(iter_book_ticker_events(path, 1.0, chunksize=2)),
        ignore_index=True,
    )
    assert events["event_type"].tolist() == [1, 2]
    assert events["jump_size_ticks"].tolist() == pytest.approx([1.0, 1.0])
    assert events["inter_arrival"].tolist() == pytest.approx([0.1, 0.1])


def test_concatenate_event_chunks_preserves_inter_arrivals() -> None:
    chunks = [
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01T00:00:00.7Z"], utc=True),
                "event_type": [1],
                "inter_arrival": [0.7],
                "jump_size_ticks": [1.0],
                "mid_price": [100.0],
            }
        ),
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-01-01T00:00:01.2Z"], utc=True),
                "event_type": [2],
                "inter_arrival": [0.5],
                "jump_size_ticks": [1.0],
                "mid_price": [99.0],
            }
        ),
    ]

    events = concatenate_event_chunks(chunks)

    assert events["inter_arrival"].tolist() == pytest.approx([0.7, 0.5])


def test_concatenate_event_chunks_discards_zero_time_events() -> None:
    chunk = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-01-01T00:00:00.000Z", "2024-01-01T00:00:00.100Z"],
                utc=True,
            ),
            "event_type": [1, 2],
            "inter_arrival": [0.0, 0.1],
            "jump_size_ticks": [1.0, 1.0],
            "mid_price": [100.0, 99.0],
        }
    )

    events = concatenate_event_chunks([chunk])

    assert events["event_type"].tolist() == [2]
    assert events["inter_arrival"].tolist() == pytest.approx([0.1])
