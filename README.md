# TG/Mestrado — Estimação neural de processos de Hawkes

Monorepo acadêmico de João Matheus Del Vecchio França Barbosa, orientado pelo Prof. Dr. Mauri
Aparecido de Oliveira (ITA).

> **Sobre esta cópia pública.** É um retrato do repositório de trabalho, que continua privado,
> em um único commit. Ficaram de fora três coisas:
> - os PDFs da bibliografia (`references/`), que não são redistribuídos; as referências estão em
>   `thesis/latex/references.bib` e em `docs/literature_matrix.md`;
> - os dados sintéticos (`data/synthetic/`), que podem ser gerados de novo com os alvos `make`;
> - os artefatos de treino e de avaliação (checkpoints e registros do MLE), que ficam nas
>   releases do repositório privado e estão disponíveis mediante pedido.
>
> Os PDFs da monografia, do artigo e dos slides estão em `thesis/build/`.

O trabalho investiga a estimação dos parâmetros de um processo de Hawkes 2D
aplicado a movimentos de preço em alta frequência. A pesquisa parte da reprodução
de Lee (2023), compara a LSTM proposta no artigo com máxima verossimilhança e,
depois, testa uma extensão baseada em Transformer/atenção em dados de criptoativos.

## Pergunta de pesquisa

Uma arquitetura com atenção consegue estimar os parâmetros do Hawkes 2D em
sequências de preços de cripto com acurácia e estabilidade iguais ou superiores à
LSTM de Lee (2023), mantendo inferência muito mais rápida que MLE?

## Marcos

1. **Dados:** comparar Binance spot e futuros perpétuos e demonstrar que os dados
   disponíveis permitem reconstruir a entrada usada por Lee.
2. **Reprodução:** implementar simulação, MLE e a LSTM do artigo, separando
   reprodução sintética de adaptação empírica.
3. **Extensão:** comparar um encoder Transformer com a LSTM sob o mesmo protocolo.

Cada marco possui um gate. Nenhum resultado posterior deve ser interpretado antes
de o gate anterior estar documentado.

## Estado (02/11/2026)

- Marcos 1 e 2 concluídos; Marco 3 concluído no sintético (H3 não suportada: a LSTM
  é pelo menos tão boa quanto o Transformer).
- E2 reproduz o padrão de Lee no ponto fixo; a vantagem das redes no MSE agregado vem
  da robustez quando β é mal identificado, não de maior precisão por trajetória.
- Etapa empírica (escopo A, [ADR 008](docs/decisions/008-escopo-empirico-e-idioma.md)):
  redes com ajuste fino em rótulos MLE de janelas reais; teste aberto uma vez (MSE 0,044
  LSTM escolhida, 0,040 Transformer, 0,218 média dos rótulos). Resultados congelados na tag
  `resultados-finais`.
- Adendo ([ADR 009](docs/decisions/009-adendo-ajuste-fino-e-beta.md)): numa semana nova
  (5–11/fev/2024), a LSTM com teto de 1.000 épocas (0,0212) e a média LSTM+Transformer
  (0,0204) vencem a LSTM escolhida (0,0264); a grade de 100 ms faz o MLE estimar ~58% do β
  contínuo.
- Escopo B reduzido ([ADR 010](docs/decisions/010-escopo-b-reduzido.md)): prior a partir dos
  rótulos reais, simulado na grade; na semana de 4–10/mar/2024 a LSTM pré-treinada nele erra
  63% menos (0,040 × 0,109).
- Correção da grade e volatilidade ([ADR 011](docs/decisions/011-correcao-da-grade-e-volatilidade.md)):
  a rede treinada com o β verdadeiro desfaz o viés da grade em simulação, mas discorda do mapa
  nas janelas reais; com o Transformer, a volatilidade de Hawkes ordena melhor as janelas.
- Texto: monografia em `thesis/tg/`, artigo em `thesis/paper/`, slides em `thesis/slides/`.

## Estrutura

```text
admin/                 normas e formulários do ITA
archive/               exemplos e material histórico
configs/               parâmetros versionados dos experimentos
data/                  raw/interim/processed/external (conteúdo não versionado)
docs/                  metodologia, decisões e protocolo de reprodução
notebooks/             exploração; resultados finais devem existir também em scripts
proposal/              proposta e apresentação de TG-1
references/            bibliografia organizada por tema
scripts/               pontos de entrada reproduzíveis
src/hawkes_crypto/     pacote Python do projeto
tests/                 testes estatísticos e de software
thesis/                estrutura e fontes da monografia
```

## Início rápido

Requer Python 3.11–3.13. O Python 3.14 ainda não é suportado por toda a pilha
científica usada no projeto.

```bash
uv venv --python 3.13
source .venv/bin/activate
uv sync --extra dev
pytest
```

Validação ponta a ponta do protocolo neural:

```bash
make smoke
```

Esse alvo usa poucas trajetórias e duas épocas. O protocolo científico final usa
exclusivamente sequências de 1.000 eventos, com 75.000/10.000/15.000 trajetórias,
até 300 épocas e cinco seeds por rede. Para reproduzir dados, treino, MLE,
consolidação, figuras e os três PDFs:

```bash
make synthetic-final
make local-final
```

As etapas são retomáveis; a execução integral pode levar horas ou dias.

Auditoria local de um CSV de `bookTicker`:

```bash
python scripts/audit_binance_data.py \
  --input data/raw/sample.csv \
  --market um_futures \
  --symbol BTCUSDT \
  --tick-size 0.10
```

## Reprodutibilidade

- Seeds, parâmetros e splits ficam em `configs/`.
- Dados brutos, processados, checkpoints e resultados pesados não entram no Git.
- Cada dataset deve possuir manifesto com fonte, URL, checksum e transformação.
- Cada resultado deve registrar commit, config, seed, versões e hardware.
- Ambiguidades do paper-base são registradas em
  [`docs/replication_ledger.md`](docs/replication_ledger.md).

## Referência principal

Lee, K. (2023). *Recurrent neural network based parameter estimation of Hawkes
model on high-frequency financial data*. Finance Research Letters, 55, 103922.
https://doi.org/10.1016/j.frl.2023.103922
