# ADR 004 — Arquitetura Transformer

## Decisão

Usar um encoder Transformer, sem decoder, como extensão E1 da LSTM de Lee. A entrada
por evento tem dois valores: `log1p(inter_arrival)` e direção codificada em `+1/-1`.
Ela é projetada de 2 para 32 dimensões e recebe encoding posicional senoidal.

O modelo final possui quatro cabeças (oito dimensões por cabeça), duas camadas,
feed-forward `32 → 128 → 32`, dropout 0,1, normalização e pooling médio temporal.
A cabeça final é compartilhada conceitualmente com a LSTM: softplus produz valores
positivos e `beta = alpha_self + alpha_cross + margin`, garantindo `g < 1`.

## Motivo

O encoder compara diretamente eventos distantes e produz uma representação global
sem exigir geração autoregressiva. Compartilhar entrada, alvo e restrição de saída
isola a arquitetura como principal diferença experimental.

## Consequências

A autoatenção custa memória e tempo quadráticos no comprimento. Por isso, o batch
do Transformer é 8 na configuração final, enquanto a LSTM usa 64. Os resultados de
1.000 eventos incluem latência e custo.
