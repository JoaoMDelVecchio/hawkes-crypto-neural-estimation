# ADR 011 — A rede corrige a grade? E a volatilidade com as redes melhores

Status: **aceita**; protocolo congelado antes de treinar e de calcular qualquer métrica
Data: 2026-11-02

## Contexto

O adendo (ADR 009) mostrou que a grade de 100 ms faz o MLE estimar cerca de 58% do β em tempo
contínuo e que o MLE em tempo contínuo não tem como desfazer isso. Uma rede treinada com o θ
verdadeiro de trajetórias levadas à grade pode aprender a corrigir esse viés: é a vantagem
estrutural da estimação amortizada sob observação discretizada que o TG deixou como trabalho
futuro (Lambe 2025). Os dados sintéticos do escopo B (ADR 010) guardam o θ verdadeiro de cada
trajetória.

O critério C4 (volatilidade) foi medido só com a rede escolhida no TG, que ordena as janelas
pior que o MLE (Spearman 0,28 × 0,78 no teste). O adendo produziu redes melhores.

Os dois são análises de adendo: o modelo escolhido no TG não muda.

## Parte 1 — correção da grade (`scripts/run_grid_correction.py`)

- **Treino:** as 20.000 + 2.000 trajetórias de `artifacts/scope_b_010/synthetic.npz` (mesma
  entrada das redes: interchegadas na grade e tipo), alvo = θ verdadeiro em tempo contínuo.
  LSTM (`lstm_lee`), 5 seeds, do zero, lr 10⁻³, lote 64, teto 150 épocas, paciência 15, MSE
  na escala original; checkpoint pelo MSE de validação sintética.
- **Teste:** 2.000 trajetórias novas do mesmo prior do ADR 010 (seed `20261105`), levadas à
  grade, cortadas em 1.000 eventos e rotuladas pelo MLE na grade (5 inícios).
- **Estimadores, todos contra o θ verdadeiro:** (a) ensemble da rede; (b) MLE na grade; (c)
  MLE corrigido: (μ̂, β̂) invertidos pelo mapa do viés da grade (`grid_bias.json`), com g e a
  fração cruzada do MLE aplicados ao β corrigido.
- **Perda por trajetória:** erro quadrático médio nos 4 parâmetros.
- **Comparação principal:** rede × MLE corrigido, pareada por trajetória; favorável se o IC
  bootstrap 95% da diferença média ficar abaixo de zero.
- **Secundárias (Holm):** rede × MLE na grade; MLE corrigido × MLE na grade.
- **Também relatados:** MSE e mediana por parâmetro; erro relativo mediano de β.
- **Exploratório:** a rede aplicada às janelas reais de validação e da semana de fevereiro
  (β contínuo implícito, comparado com a inversão do mapa).
- **Ressalva registrada antes:** treino e teste vêm do mesmo prior, o que favorece a rede (ela
  aproxima a média a posteriori sob esse prior); o mapa do viés foi medido só com g = 0,5 e
  α_c ≈ 0, o que desfavorece o MLE corrigido fora disso.

## Parte 2 — volatilidade (`scripts/run_volatility_addendum.py`)

- **Janelas:** as 711 da semana de fevereiro (já lida no ADR 009; análise *post hoc*) e as
  476 de `validation`.
- **Estimadores:** MLE da janela; `lstm_lee/finetune` (congelado); `lstm_lee/finetune_long`;
  `transformer/finetune`; `mix/lstm+transformer`.
- **Medidas:** volatilidade de Hawkes pela fórmula do C4 contra a realizada com Δ = 10 s:
  Spearman entre janelas e razão mediana.
- **Comparação principal:** Spearman da mistura − Spearman da LSTM congelada, na semana de
  fevereiro, com IC bootstrap 95% (2.000 reamostragens das janelas).

## Consequências

Os resultados entram no adendo do texto qualquer que seja o sinal. Teto: 4 h de máquina.

## Resultado da parte 2 (volatilidade)

Δ = 10 s; Spearman entre a volatilidade de Hawkes e a realizada, e razão mediana.

| Estimador | Fevereiro (711) | Razão | Validação (476) | Razão |
|---|---|---|---|---|
| MLE | 0,765 | 1,35 | 0,667 | 1,24 |
| `lstm_lee/finetune` (congelado) | 0,471 | 1,41 | 0,468 | 1,24 |
| `lstm_lee/finetune_long` | 0,494 | 1,39 | 0,501 | 1,23 |
| `transformer/finetune` | 0,688 | 1,31 | 0,608 | 1,20 |
| `mix/lstm+transformer` | 0,626 | 1,36 | 0,586 | 1,22 |

**Principal favorável:** mistura − LSTM congelada = +0,155 (IC 95% [0,130; 0,181]). O nível
fica certo com todos os estimadores; eles diferem na ordenação. Treinar a LSTM por mais
tempo quase não melhora a ordenação (+0,02); o Transformer, sim, e é a rede que mais se
aproxima do MLE. A LSTM erra menos no MSE dos parâmetros, mas o Transformer preserva melhor as
diferenças entre janelas, que é o que a volatilidade exige.

## Resultado da parte 1 (correção da grade)

5 seeds, melhor época 134–148 de 150 (o teto quase foi atingido). 2.000 trajetórias de teste,
todas com MLE convergido; a inversão cai dentro do mapa em 93,5% delas.

| Estimador | MSE macro | MSE μ | MSE α_s | MSE β | Viés mediano β | Erro abs. mediano β |
|---|---|---|---|---|---|---|
| Rede (alvo = θ verdadeiro) | **0,328** | 0,035 | 0,290 | 0,899 | +2,8% | 8,9% |
| MLE na grade | 3,951 | 0,012 | 4,062 | 11,680 | −44,1% | 44,1% |
| MLE corrigido pelo mapa | 0,654 | 0,011 | 0,993 | 1,589 | −2,4% | 8,4% |

**Principal favorável:** rede − MLE corrigido = −0,326 (IC 95% [−0,405; −0,256], Wilcoxon
p ≈ 10⁻³⁵); a rede erra menos em 60% das trajetórias. **Secundárias (Holm):** rede − MLE na
grade = −3,62 (IC [−3,91; −3,36]), a rede ganha em 95%; corrigido − MLE na grade = −3,30
(IC [−3,53; −3,07]), o corrigido ganha em 99%.

**Leitura.** A correção pelo mapa, sozinha, já remove quase todo o viés de β (−44% → −2%) e
tem o menor erro mediano em β e em μ. A rede ganha na média sobretudo em α_s e nas caudas de β
(MSE de β 0,90 × 1,59), onde o mapa, medido só com g = 0,5, extrapola mal. Os dois corrigem a
grade: o mapa tem o menor erro mediano, e a rede erra menos nas caudas.

**Exploratório — janelas reais.** A rede implica β contínuo mediano de 7,7 na validação e
6,3 em fevereiro, contra 4,9 e 3,1 da inversão do mapa e 2,76 e 2,16 no rótulo da grade;
Spearman rede × inversão de 0,55 nas duas. A rede leva β para muito acima do que o mapa
indica. Como as janelas reais não são trajetórias de Hawkes (resíduos com dependência,
saltos de vários ticks), a correção aprendida em dados simulados não se transfere com
segurança: dentro do prior a rede é a melhor correção; nos dados reais, os dois métodos
discordam e nenhum pode ser verificado sem o tempo contínuo verdadeiro.
