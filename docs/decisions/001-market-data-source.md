# ADR 001 — Mercado e fonte de dados

Status: **executada e aprovada**
Data: 2026-08-31

## Contexto

Lee (2023) usa NBBO para reconstruir o mid-price observado a cada 0,1 s. Para uma
adaptação válida, precisamos de melhor bid/ask histórico, não apenas negócios ou candles.

## Evidência inicial

Em 2026-08-31, a documentação oficial do `binance-public-data` e testes diretos para
2024-01-15 apresentaram:

| Mercado/produto | Endpoint testado | Resultado |
|---|---|---|
| USD-M `bookTicker` | `data/futures/um/daily/bookTicker/BTCUSDT/...` | HTTP 200 |
| Spot `bookTicker` | `data/spot/daily/bookTicker/BTCUSDT/...` | HTTP 404 |
| USD-M `aggTrades` | `data/futures/um/daily/aggTrades/BTCUSDT/...` | HTTP 200 |
| Spot `aggTrades` | `data/spot/daily/aggTrades/BTCUSDT/...` | HTTP 200 |

O arquivo USD-M de `bookTicker` desse único dia possui aproximadamente 188 MB
compactado, o que também exige um piloto curto antes de definir o período.

O piloto com as primeiras 1.000.000 de linhas de BTCUSDT em 2024-01-15 cobriu
3.156,117 segundos e apresentou:

- 4.850 mudanças do mid-price após amostragem em 100 ms;
- 4 segmentos não sobrepostos de 1.000 eventos;
- zero cotações cruzadas, preços inválidos e inversões temporais;
- intervalo mediano de 2 ms entre timestamps distintos e p99 de 117 ms;
- 713.128 timestamps repetidos, tratados mantendo o último estado do instante.

Essa densidade é compatível com a construção das sequências do paper. A frequência de
timestamps repetidos será monitorada, mas não representa por si só dados duplicados:
atualizações distintas podem compartilhar o mesmo timestamp de evento.

## Decisão

1. **Futuros perpétuos USD-M** são o candidato primário porque oferecem os dois fluxos
   necessários no acervo público.
2. Spot não é comparável usando exclusivamente o acervo oficial: sem `bookTicker`,
   negócios não permitem reconstruir o NBBO/mid-price exigido.
3. A fonte/mercado está aprovada para a pipeline; ETHUSDT e o período definitivo foram
   auditados na construção do dataset principal.
4. Uma comparação spot futura exigiria outra fonte histórica de top-of-book, que teria
   de ser avaliada separadamente por custo, licença e qualidade.

## Consequências

- O downloader rejeita `bookTicker` para spot explicitamente.
- Dados brutos não serão versionados.
- O período completo só será escolhido após estimativa de volume e qualidade no piloto.
