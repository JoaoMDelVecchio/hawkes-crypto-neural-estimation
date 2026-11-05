# Especificação e gate dos dados

Status: **gate executado e aprovado para futuros USD-M, BTCUSDT e ETHUSDT**.

## Candidatos

- Fonte principal: Binance Public Data (`data.binance.vision`).
- Mercados: spot e futuros perpétuos USDT-M.
- Símbolos: BTCUSDT e ETHUSDT.
- Produtos desejados: melhor bid/ask (`bookTicker`), negócios agregados
  (`aggTrades`) e metadados do instrumento/tick size.

O inventário confirmou `bookTicker` e `aggTrades` em futuros USD-M. Spot oferece
`aggTrades`, mas não o histórico oficial de `bookTicker`, portanto não atende sozinho
ao requisito de reconstrução do mid-price.

## Schema canônico de cotação

| Campo | Tipo | Regra |
|---|---|---|
| `timestamp` | UTC datetime/ns | não nulo e ordenável |
| `bid_price` | float64 | positivo |
| `bid_qty` | float64 | não negativo |
| `ask_price` | float64 | positivo e `ask_price >= bid_price` |
| `ask_qty` | float64 | não negativo |
| `symbol` | string | BTCUSDT ou ETHUSDT |
| `market` | category | `spot` ou `um_futures` |

## Schema canônico de eventos

| Campo | Tipo | Regra |
|---|---|---|
| `timestamp` | UTC datetime/ns | instante da observação em grade |
| `event_type` | int8 | alta = 1; baixa = 2 |
| `inter_arrival` | float64 | segundos, estritamente positivo |
| `jump_size_ticks` | float64 | módulo do salto dividido pelo tick |
| `mid_price` | float64 | preço após a mudança |

## Critérios de escolha

1. Cobertura e estabilidade da fonte.
2. Timestamps e best bid/ask suficientes para grade de 100 ms.
3. Poucos gaps/erros não explicados.
4. Liquidez e número de segmentos válidos.
5. Tick size recuperável por data.
6. Reprodutibilidade e licença de uso acadêmico.

## Artefatos do piloto

- `data/raw/<market>/<symbol>/<date>/...`
- `data/raw/manifest.jsonl` com URL, bytes e SHA-256.
- `artifacts/data_audit/<run-id>/summary.json`
- decisão em `docs/decisions/001-market-data-source.md`

## Resultado inicial

A amostra de 1.000.000 de atualizações de BTCUSDT em 2024-01-15 gerou 4.850 eventos
em 52,6 minutos. O schema, a ordenação e os preços passaram pela auditoria. O CSV diário
completo possui cerca de 19,9 milhões de linhas e 1,7 GB descompactado; por isso, o
processamento definitivo é incremental e não carrega um dia inteiro em memória.
