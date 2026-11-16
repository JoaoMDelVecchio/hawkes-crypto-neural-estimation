"""Figura dos slides: eventos reais do bloco de calibração contra um Poisson de mesma taxa.

Usa um único dia do bloco de calibração (BTCUSDT, 15 jan. 2024), nunca o bloco de teste. À
esquerda, dois minutos de eventos reais e de um Poisson homogêneo com a mesma taxa média; à
direita, o índice de dispersão (variância / média das contagens) por tamanho de janela, que é
1 para um Poisson. Figura ilustrativa dos slides: não muda nenhum número do TG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = [0.5, 1, 2, 5, 10, 30, 60, 120]


def dispersion(times: np.ndarray, width: float) -> float:
    counts = np.bincount((times // width).astype(int))
    return float(counts.var() / counts.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--day", type=Path,
                        default=Path("data/processed/um_futures/BTCUSDT/2024-01-15.parquet"))
    parser.add_argument("--start", type=float, default=78_480.0,
                        help="segundos desde o início (taxa média do dia, rajadas visíveis)")
    parser.add_argument("--span", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20261116)
    parser.add_argument("--output", type=Path,
                        default=Path("thesis/build/figures/slides_real_vs_poisson.pdf"))
    args = parser.parse_args()

    day = pd.read_parquet(args.day)
    times = (day.timestamp - day.timestamp.iloc[0]).dt.total_seconds().to_numpy()
    types = day.event_type.to_numpy()
    rate = len(times) / times[-1]

    rng = np.random.default_rng(args.seed)
    poisson = np.cumsum(rng.exponential(1.0 / rate, size=int(rate * times[-1] * 1.1)))
    poisson = poisson[poisson < times[-1]]

    lo, hi = args.start, args.start + args.span
    sel = (times >= lo) & (times < hi)
    real_t, real_c = times[sel] - lo, types[sel]
    pois_t = poisson[(poisson >= lo) & (poisson < hi)] - lo

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2.2, 1]})
    up, down = real_t[real_c == 1], real_t[real_c == 2]
    ax.vlines(up, 1.55, 1.95, color="#2E7D32", lw=0.8, label="up-move")
    ax.vlines(down, 1.05, 1.45, color="#C8553D", lw=0.8, label="down-move")
    ax.vlines(pois_t, 0.05, 0.45, color="#1F2A44", lw=0.8)
    ax.set_yticks([1.5, 0.25], ["BTCUSDT\n(real)", "Poisson,\nsame rate"])
    ax.set_xlim(0, args.span)
    ax.set_ylim(-0.05, 2.05)
    ax.set_xlabel("time (s)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.2), ncol=2, frameon=False, fontsize=9)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    bx.plot(BINS, [dispersion(times, b) for b in BINS], "o-", color="#C8553D", label="real")
    bx.plot(BINS, [dispersion(poisson, b) for b in BINS], "o-", color="#1F2A44", label="Poisson")
    bx.set_xscale("log")
    bx.set_xticks([1, 10, 60], ["1 s", "10 s", "60 s"])
    bx.set_xlabel("counting window")
    bx.set_ylabel("variance / mean of counts")
    bx.legend(frameon=False, fontsize=9)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    print(f"taxa {rate:.3f}/s; janela {sel.sum()} reais, {len(pois_t)} Poisson; dispersão 60 s: "
          f"real {dispersion(times, 60):.1f}, Poisson {dispersion(poisson, 60):.2f}")
    print(f"salvo em {args.output}")


if __name__ == "__main__":
    main()
