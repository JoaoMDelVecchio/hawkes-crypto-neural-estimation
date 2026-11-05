# Protocolo experimental

## Princípios

- Separar reprodução sintética, adaptação empírica e extensão.
- Fixar o protocolo antes de abrir o conjunto de teste.
- Fazer splits cronológicos por blocos; janelas sobrepostas pertencem ao mesmo split.
- Comparar modelos com a mesma entrada, alvo, loss e orçamento declarado.

## Reprodutibilidade por execução

Registrar:

- commit e arquivo de configuração;
- seed de Python, NumPy e PyTorch;
- origem/checksum dos dados e versão da transformação;
- versões de Python, bibliotecas, sistema e hardware;
- tempo de treino, inferência unitária e em lote;
- parâmetros do modelo e número de pesos treináveis.

## Experimentos mínimos

### E1 — recuperação sintética

Parâmetros conhecidos; comparar MLE, LSTM e Transformer por parâmetro e por região de
`g`, incluindo região próxima da criticidade sem atingir `g >= 1`.

### E2 — ponto fixo do artigo

`μ=0,3`, `α_self=0,4`, `α_cross=0,7`, `β=1,5`, repetido com seeds independentes.

Executado em 30/10/2026 com os checkpoints congelados (`make fixed-point`).

### E3 — adaptação empírica

MLE como alvo em segmentos reais fora do conjunto usado para calibrar as faixas.

### E4 — robustez

Separar símbolo, mercado/período e regimes de volatilidade.

## Métricas primárias

- MSE macro e por parâmetro;
- erro relativo mediano;
- viés;
- proporção de estimativas válidas;
- latência mediana e p95.

O Transformer só será declarado superior se melhorar a métrica primária no teste
congelado sem piora material de validade e estabilidade.

Métricas e testes complementares (ADR 007): R² por parâmetro em relação à média a
priori, erro absoluto mediano, comparação pareada por trajetória com IC *bootstrap* e
testes entre seeds. Diferenças entre redes só são afirmadas com significância nos dois
níveis ou com a ressalva explícita.

## Protocolo congelado de produção

A configuração normativa é `configs/lee_synthetic_final.yaml`. São geradas
trajetórias independentes de 1.000 eventos: 75.000 para treino, 10.000 para validação
e 15.000 para teste. A amostragem é balanceada em três regiões de `g`: baixa
(`0,10--0,35`), intermediária (`0,35--0,70`) e próxima da criticidade
(`0,70--0,95`). Cada shard possui manifesto e SHA-256.

LSTM e Transformer são avaliados em cinco seeds independentes. Para os estimadores
neurais, a melhor época é escolhida exclusivamente pelo MSE da validação, com máximo
de 300 épocas e paciência 30. Checkpoints, histórico, configuração resolvida,
ambiente e commit são persistidos. O MLE é ajustado separadamente em cada uma das
15.000 trajetórias de teste, usa a taxa empírica como inicialização heurística e
múltiplos starts aleatórios; ele nunca recebe o parâmetro verdadeiro como
inicialização.

`configs/lee_local_optimized.yaml` é operacionalmente equivalente e reutiliza os
mesmos dados. Ela adiciona batching seguro, gradient accumulation, compilação onde
aprovada por benchmark e checkpoints retomáveis, sem mudar o batch efetivo ou os
hiperparâmetros científicos. O perfil congelado está em
`configs/local_execution_profile.json`.

## Política dos dados empíricos

- `calibration_train`: 15--22 de janeiro de 2024; único bloco permitido para ajuste;
- `validation`: 23--25 de janeiro; seleção de época/configuração;
- `test_closed`: 26--28 de janeiro; requer a flag explícita `--open-test`;
- robustez: 23--29 de outubro de 2023; checkpoint congelado e nenhum retreino.

Nos dados reais, o alvo é o ajuste MLE da janela. Portanto, o erro mede aproximação
amortizada do MLE, não recuperação de um parâmetro verdadeiro de mercado.

## Execução escalonada

```bash
# integração local no comprimento final de 1.000 eventos
make smoke

# carga integral; executar somente com orçamento de horas/dias
make synthetic-final
make train-final

# execução local otimizada e retomável
make benchmark-local
make local-final

# calibração sem abrir o teste
make calibration
```

O teste empírico e a robustez são executados somente depois de congelar o checkpoint.
`make empirical-test` abre o teste de forma deliberada. A robustez exige
`MODEL=... CHECKPOINT=...` e usa `LENGTH=1000`.
