# Plano de implementação

## Marco 1 — dados adequados ao problema

Objetivo: escolher por evidência entre Binance spot e futuros perpétuos USDT-M para
BTCUSDT e ETHUSDT.

1. Inventariar `bookTicker`, `aggTrades` e metadados dos instrumentos.
2. Baixar uma amostra curta de cada mercado com URL, data e checksum.
3. Auditar schema, ordenação, duplicatas, gaps, spread e continuidade.
4. Reconstruir mid-price em grade de 0,1 s.
5. Gerar eventos com `inter_arrival`, `event_type` e `jump_size_ticks`.
6. Comparar cobertura, liquidez e quantidade de segmentos de 1.000 eventos.
7. Registrar a escolha em `docs/decisions/`.

Gate: fonte, mercado, símbolos, intervalo e contrato de dados aprovados.

## Marco 2 — reprodução de Lee (2023)

1. Registrar especificações e lacunas em `replication_ledger.md`.
2. Implementar Hawkes 2D simétrico e simulador por thinning de Ogata.
3. Implementar MLE com restrições, múltiplas inicializações e diagnósticos.
4. Calibrar faixas dos parâmetros a partir dos dados escolhidos.
5. Gerar treino sintético com seed e manifesto.
6. Reproduzir duas camadas LSTM de 12 unidades e saída para quatro parâmetros.
7. Comparar LSTM e MLE em simulação e, separadamente, em dados reais.

Gate: pipeline reproduzível e conclusão documentada sobre o grau de reprodução.

## Marco 3 — Transformer/atenção

1. Fixar dados, splits, comprimento, loss, restrições, métricas e orçamento.
2. Implementar encoder temporal compacto com pooling para regressão.
3. Fazer ablações mínimas de encoding, pooling, camadas e cabeças.
4. Comparar Transformer, LSTM e MLE no teste mantido fora do ajuste.
5. Avaliar acurácia, viés, estabilidade, latência, memória e robustez por regime.

Gate: aceitar ou rejeitar a hipótese de superioridade da atenção sem alterar o
protocolo após observar o teste.

## Aplicação

O melhor estimador será usado para calcular volatilidade Hawkes intradiária, comparada
à realized volatility. Essa análise é posterior e não substitui o teste principal de
estimação.
