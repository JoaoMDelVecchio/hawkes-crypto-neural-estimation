# ADR 002 — Janela primária

Status: **aprovada**  
Data: 2026-08-31

## Decisão

Usar Binance USD-M Futures, BTCUSDT e ETHUSDT, de 2024-01-15 a 2024-01-28 UTC,
com `bookTicker` amostrado em 100 ms e `aggTrades` como validação auxiliar.

## Evidência executada

Todos os 56 arquivos esperados (14 dias × 2 símbolos × 2 produtos) retornaram HTTP
200. O processamento gerou 28 partições Parquet:

- BTCUSDT: 887.803 eventos; mínimo diário de 16.576; 880 segmentos de 1.000;
- ETHUSDT: 939.101 eventos; mínimo diário de 20.828; 933 segmentos de 1.000;
- total: 1.826.904 eventos;
- 1.813 janelas de 1.000 eventos no índice vigente.

Não houve ausência de arquivo no período. A seleção é suficiente para reprodução,
validação e teste cronológico, mas continua sendo adaptação empírica porque Lee (2023)
usou NBBO de ações e não disponibilizou o acervo original.

## Split congelado

- calibração/treino: 15–22 jan;
- validação: 23–25 jan;
- teste fechado: 26–28 jan.
