"""Normalização, auditoria e construção dos eventos de mid-price."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "update_id",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
    "transaction_time",
    "event_time",
]

ALIASES = {
    "u": "update_id",
    "updateId": "update_id",
    "bidPrice": "bid_price",
    "best_bid_price": "bid_price",
    "b": "bid_price",
    "bidQty": "bid_qty",
    "best_bid_qty": "bid_qty",
    "B": "bid_qty",
    "askPrice": "ask_price",
    "best_ask_price": "ask_price",
    "a": "ask_price",
    "askQty": "ask_qty",
    "best_ask_qty": "ask_qty",
    "A": "ask_qty",
    "transaction_time": "transaction_time",
    "T": "transaction_time",
    "event_time": "event_time",
    "E": "event_time",
}


@dataclass(frozen=True)
class QuoteAudit:
    rows: int
    duplicate_timestamps: int
    out_of_order: int
    invalid_prices: int
    crossed_quotes: int
    zero_quantities: int
    median_update_ms: float | None
    p99_update_ms: float | None
    duration_seconds: float

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _timestamp_unit(values: pd.Series) -> str:
    median = float(pd.to_numeric(values, errors="coerce").dropna().median())
    if median >= 1e17:
        return "ns"
    if median >= 1e14:
        return "us"
    if median >= 1e11:
        return "ms"
    return "s"


def _normalize_book_ticker_frame(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = frame.rename(columns=ALIASES)
    time_column = next(
        (
            column
            for column in ("event_time", "transaction_time", "timestamp")
            if column in renamed.columns
        ),
        None,
    )
    if time_column is None:
        raise ValueError("bookTicker sem event_time/transaction_time")

    result = renamed.rename(columns={time_column: "timestamp"}).copy()
    for column in ("bid_price", "bid_qty", "ask_price", "ask_qty"):
        if column not in result:
            raise ValueError(f"Coluna obrigatória ausente: {column}")
        result[column] = pd.to_numeric(result[column], errors="coerce")

    raw_time = pd.to_numeric(result["timestamp"], errors="coerce")
    result["timestamp"] = pd.to_datetime(
        raw_time, unit=_timestamp_unit(raw_time), utc=True, errors="coerce"
    )
    return result[
        ["timestamp", "bid_price", "bid_qty", "ask_price", "ask_qty"]
    ].dropna()


def read_book_ticker_csv(path: Path, *, nrows: int | None = None) -> pd.DataFrame:
    """Lê variantes com ou sem cabeçalho do bookTicker USD-M."""
    frame = pd.read_csv(path, nrows=nrows)
    if not {"bid_price", "ask_price"}.issubset(frame.rename(columns=ALIASES).columns):
        frame = pd.read_csv(path, header=None, nrows=nrows)
        if frame.shape[1] != len(CANONICAL_COLUMNS):
            raise ValueError(
                f"Schema bookTicker desconhecido: {frame.shape[1]} colunas em {path}"
            )
        frame.columns = CANONICAL_COLUMNS
    return _normalize_book_ticker_frame(frame)


def audit_quotes(quotes: pd.DataFrame) -> QuoteAudit:
    """Resume falhas que podem invalidar a reconstrução do mid-price."""
    required = {"timestamp", "bid_price", "bid_qty", "ask_price", "ask_qty"}
    missing = required.difference(quotes.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")

    timestamps = pd.to_datetime(quotes["timestamp"], utc=True)
    deltas_ms = timestamps.diff().dt.total_seconds().mul(1000)
    positive_deltas = deltas_ms[deltas_ms > 0]
    duration = (
        float((timestamps.max() - timestamps.min()).total_seconds())
        if len(timestamps) > 1
        else 0.0
    )
    return QuoteAudit(
        rows=len(quotes),
        duplicate_timestamps=int(timestamps.duplicated().sum()),
        out_of_order=int((deltas_ms < 0).sum()),
        invalid_prices=int(
            ((quotes["bid_price"] <= 0) | (quotes["ask_price"] <= 0)).sum()
        ),
        crossed_quotes=int((quotes["bid_price"] > quotes["ask_price"]).sum()),
        zero_quantities=int(
            ((quotes["bid_qty"] <= 0) | (quotes["ask_qty"] <= 0)).sum()
        ),
        median_update_ms=(
            float(positive_deltas.median()) if not positive_deltas.empty else None
        ),
        p99_update_ms=(
            float(positive_deltas.quantile(0.99))
            if not positive_deltas.empty
            else None
        ),
        duration_seconds=duration,
    )


def quotes_to_events(
    quotes: pd.DataFrame,
    tick_size: float,
    *,
    frequency: str = "100ms",
) -> pd.DataFrame:
    """Transforma top-of-book na sequência usada pelos estimadores."""
    if tick_size <= 0:
        raise ValueError("tick_size deve ser positivo")
    audit = audit_quotes(quotes)
    if audit.invalid_prices or audit.crossed_quotes:
        raise ValueError("Cotações inválidas; consulte audit_quotes antes de limpar")

    ordered = quotes.sort_values("timestamp").drop_duplicates(
        subset="timestamp", keep="last"
    )
    indexed = ordered.set_index(pd.DatetimeIndex(ordered["timestamp"]))
    grid = indexed[["bid_price", "ask_price"]].resample(frequency).last().ffill()
    grid["mid_price"] = (grid["bid_price"] + grid["ask_price"]) / 2.0
    grid["delta"] = grid["mid_price"].diff()
    events = grid.loc[grid["delta"].ne(0) & grid["delta"].notna()].copy()
    events["event_type"] = np.where(events["delta"] > 0, 1, 2).astype("int8")
    events["jump_size_ticks"] = events["delta"].abs() / tick_size
    inter_arrivals = events.index.to_series().diff().dt.total_seconds()
    if not events.empty:
        inter_arrivals.iloc[0] = (
            events.index[0] - grid.index[0]
        ).total_seconds()
    events["inter_arrival"] = inter_arrivals.astype("float64")
    events = events.loc[events["inter_arrival"].gt(0)]
    events.index.name = "timestamp"
    return events[
        ["event_type", "inter_arrival", "jump_size_ticks", "mid_price"]
    ].reset_index()


def iter_book_ticker_events(
    path: Path,
    tick_size: float,
    *,
    frequency: str = "100ms",
    chunksize: int = 1_000_000,
) -> Iterator[pd.DataFrame]:
    """Processa arquivos diários grandes sem carregá-los integralmente.

    O último bucket de cada chunk é retido para que múltiplas atualizações no limite
    entre chunks sejam reduzidas ao estado final correto.
    """
    if tick_size <= 0:
        raise ValueError("tick_size deve ser positivo")
    pending = pd.DataFrame()
    previous_mid: float | None = None
    previous_event_time: pd.Timestamp | None = None
    observation_start: pd.Timestamp | None = None

    def convert_bins(bins: pd.DataFrame) -> pd.DataFrame:
        nonlocal previous_mid, previous_event_time, observation_start
        if bins.empty:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "event_type",
                    "inter_arrival",
                    "jump_size_ticks",
                    "mid_price",
                ]
            )
        bins = bins.sort_values("grid_time")
        timestamps = pd.DatetimeIndex(bins["grid_time"])
        mids = ((bins["bid_price"] + bins["ask_price"]) / 2.0).to_numpy()
        if observation_start is None:
            observation_start = timestamps[0]
        prior = np.r_[mids[0] if previous_mid is None else previous_mid, mids[:-1]]
        deltas = mids - prior
        changed = ~np.isclose(deltas, 0.0, rtol=0.0, atol=tick_size * 1e-9)
        previous_mid = float(mids[-1])
        if not changed.any():
            return pd.DataFrame()

        event_times = timestamps[changed]
        anchor = previous_event_time or observation_start
        timeline = pd.DatetimeIndex([anchor]).append(event_times).as_unit("ns")
        inter_arrivals = np.diff(timeline.asi8).astype(np.float64) / 1e9
        previous_event_time = event_times[-1]
        return pd.DataFrame(
            {
                "timestamp": event_times,
                "event_type": np.where(deltas[changed] > 0, 1, 2).astype("int8"),
                "inter_arrival": inter_arrivals,
                "jump_size_ticks": np.abs(deltas[changed]) / tick_size,
                "mid_price": mids[changed],
            }
        )

    for raw_chunk in pd.read_csv(path, chunksize=chunksize):
        normalized = _normalize_book_ticker_frame(raw_chunk)
        audit = audit_quotes(normalized)
        if audit.invalid_prices or audit.crossed_quotes or audit.out_of_order:
            raise ValueError("Chunk contém cotações inválidas ou fora de ordem")
        normalized["grid_time"] = normalized["timestamp"].dt.floor(frequency)
        bins = (
            normalized.sort_values("timestamp")
            .groupby("grid_time", as_index=False)
            .last()
        )
        if not pending.empty:
            bins = (
                pd.concat([pending, bins], ignore_index=True)
                .sort_values("timestamp")
                .groupby("grid_time", as_index=False)
                .last()
            )
        pending = bins.tail(1)
        completed = bins.iloc[:-1]
        events = convert_bins(completed)
        if not events.empty:
            yield events

    events = convert_bins(pending)
    if not events.empty:
        yield events


def concatenate_event_chunks(chunks: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatena chunks sem recalcular as interchegadas geradas pelo conversor."""
    if not chunks:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "event_type",
                "inter_arrival",
                "jump_size_ticks",
                "mid_price",
            ]
        )
    events = pd.concat(chunks, ignore_index=True)
    if (~np.isfinite(events["inter_arrival"])).any():
        raise ValueError("Chunks contêm interchegadas não finitas")
    return events.loc[events["inter_arrival"].gt(0)].reset_index(drop=True)
