# TG-2 — Hawkes Process Estimation on High-Frequency Crypto Data via Neural Networks

Código e experimentos do TG-2 de João Matheus Del Vecchio (ITA, Eng. Computação),
orientação do Prof. Dr. Mauri Aparecido de Oliveira. Continuação da proposta aceita como
TG-1: replicar Lee (2023) — estimação de parâmetros de processos de Hawkes por LSTM — e
estendê-la a dados de criptomoedas (BTCUSDT, ETHUSDT), comparando LSTM, GRU,
LSTM profunda e um encoder com atenção (Transformer), com aplicação à volatilidade de
Hawkes intradiária.

O plano de trabalho detalhado está em `plano_implementacao.md`.

## Estrutura

```
src/
  data/         # download e limpeza dos dumps Binance (bookTicker, aggTrades)
  simulate/     # motor de simulação Hawkes (Ogata thinning) e gerador de dataset sintético
  estimate/     # baseline MLE (log-verossimilhança, otimização numérica)
  models/       # arquiteturas de rede: LSTM (referência), GRU, LSTM profunda, Transformer
  eval/         # métricas de acurácia (MSE vs MLE), custo computacional, robustez
  volatility/   # aplicação: volatilidade de Hawkes intradiária vs realized volatility
notebooks/      # exploração e geração de figuras para a monografia
data/
  raw/          # dumps brutos baixados do Binance Public Data (não versionar)
  processed/    # séries de eventos (tempos, marcas) já limpas
configs/        # arquivos de configuração (janelas de data, hiperparâmetros, seeds)
tests/          # testes unitários (simulador, MLE, pipeline de dados)
scripts/        # scripts de linha de comando (download, treino, avaliação)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dados: `python scripts/download_binance_data.py --symbol BTCUSDT --product bookTicker --start 2024-01-01 --end 2024-01-31`
(o script grava em `data/raw/`; ver `plano_implementacao.md` §4 para a janela definitiva).

## Reprodutibilidade

Seeds fixas em `configs/`, checkpoints de modelo versionados por hash de config, e todo
experimento reportado registra: janela de dados, seed, arquitetura, número de parâmetros,
tempo de treino/inferência e métricas (ver `src/eval/`).
