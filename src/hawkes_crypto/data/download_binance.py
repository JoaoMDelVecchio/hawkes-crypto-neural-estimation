"""Download reproduzível dos arquivos públicos da Binance."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import requests

Market = Literal["spot", "um_futures"]
Product = Literal["bookTicker", "aggTrades"]

BASE_URL = "https://data.binance.vision/data"
MARKET_PATH = {"spot": "spot", "um_futures": "futures/um"}
SUPPORTED_PRODUCTS: dict[Market, set[Product]] = {
    # O acervo oficial não publica bookTicker histórico para spot.
    "spot": {"aggTrades"},
    "um_futures": {"bookTicker", "aggTrades"},
}


@dataclass(frozen=True)
class DownloadRecord:
    """Metadados suficientes para identificar e verificar um arquivo bruto."""

    url: str
    path: str
    market: Market
    product: Product
    symbol: str
    date: str
    bytes: int
    sha256: str


def daily_archive_url(
    market: Market, product: Product, symbol: str, day: date
) -> str:
    """Constrói a URL oficial de um arquivo diário."""
    if market not in MARKET_PATH:
        raise ValueError(f"Mercado inválido: {market}")
    if product not in SUPPORTED_PRODUCTS[market]:
        raise ValueError(f"{product} não está disponível para {market}")
    normalized_symbol = symbol.upper()
    stamp = day.isoformat()
    filename = f"{normalized_symbol}-{product}-{stamp}.zip"
    return (
        f"{BASE_URL}/{MARKET_PATH[market]}/daily/{product}/"
        f"{normalized_symbol}/{filename}"
    )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula SHA-256 sem carregar o arquivo inteiro em memória."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_daily_archive(
    market: Market,
    product: Product,
    symbol: str,
    day: date,
    output_root: Path,
    *,
    timeout: tuple[float, float] = (10.0, 120.0),
) -> DownloadRecord:
    """Baixa um ZIP diário de forma atômica e retorna seu manifesto."""
    url = daily_archive_url(market, product, symbol, day)
    destination = (
        output_root
        / market
        / symbol.upper()
        / day.isoformat()
        / Path(url).name
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    if not destination.exists():
        with requests.get(url, stream=True, timeout=timeout) as response:
            if response.status_code == 404:
                raise FileNotFoundError(f"Arquivo não publicado: {url}")
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
        if not zipfile.is_zipfile(partial):
            partial.unlink(missing_ok=True)
            raise ValueError(f"Resposta não é um ZIP válido: {url}")
        partial.replace(destination)

    return DownloadRecord(
        url=url,
        path=str(destination),
        market=market,
        product=product,
        symbol=symbol.upper(),
        date=day.isoformat(),
        bytes=destination.stat().st_size,
        sha256=sha256_file(destination),
    )


def extract_archive(record: DownloadRecord, output_dir: Path) -> list[Path]:
    """Extrai um arquivo verificado, recusando caminhos fora do destino."""
    archive = Path(record.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zipped:
        root = output_dir.resolve()
        for member in zipped.infolist():
            target = (output_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise ValueError(f"Caminho inseguro no ZIP: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
            extracted.append(target)
    return extracted


def append_manifest(record: DownloadRecord, manifest_path: Path) -> None:
    """Acrescenta um registro JSONL ao manifesto local."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(record), ensure_ascii=False, sort_keys=True))
        handle.write("\n")
