# Metodologia

## Objeto

Cada alteração do mid-price gera um evento de tipo 1 (alta) ou 2 (baixa). O tempo
entre eventos e o tipo formam a entrada principal dos estimadores. O tamanho do salto
em ticks é preservado para a aplicação de volatilidade, embora Lee (2023) assuma que
a intensidade não depende da marca.

## Hawkes 2D simétrico

Para `i,j ∈ {1,2}`:

```text
λ_i(t) = μ + Σ_j Σ_{t_k^j < t} α_ij exp[-β(t - t_k^j)]

α_11 = α_22 = α_self
α_12 = α_21 = α_cross
```

O vetor estimado é `θ = (μ, α_self, α_cross, β)`. São exigidos:

- `μ, α_self, α_cross, β > 0`;
- `g = (α_self + α_cross) / β < 1`.

## Dados observados

1. Ordenar atualizações de melhor bid/ask por timestamp.
2. Calcular `mid = (bid + ask) / 2`.
3. Observar o último estado disponível em uma grade de 100 ms.
4. Descartar pontos sem mudança do mid-price.
5. Codificar direção, interchegada e salto em ticks.
6. Dividir por tempo; nunca permitir que janelas sobrepostas cruzem splits.

Spot e futuros serão comparados antes da escolha. Não se assume antecipadamente que
um produto exista ou possua cobertura histórica adequada.

## Reprodução

A reprodução possui duas camadas:

- **Sintética:** parâmetros conhecidos, simulação e comparação LSTM versus MLE.
- **Empírica:** segmentos de cripto rotulados pelo MLE e inferência amortizada.

Isso evita confundir a reprodução do estimador com a adaptação a um mercado cujo
processo gerador não é Hawkes exato.

## Avaliação

- MSE e erro absoluto por parâmetro;
- viés e dispersão em parâmetros fixos;
- violações de positividade/estabilidade;
- convergência e sensibilidade à inicialização do MLE;
- latência por sequência e throughput em lote;
- robustez por símbolo, período e regime.

O conjunto de teste permanece fechado até o congelamento das decisões de modelagem.
