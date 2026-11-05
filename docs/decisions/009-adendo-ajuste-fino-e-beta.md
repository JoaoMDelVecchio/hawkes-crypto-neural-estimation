# ADR 009 — Adendo: teste de β, receita de ajuste fino e semana nova

Status: **aceita**; protocolo congelado antes de baixar a semana nova e de treinar
Data: 2026-11-02

## Contexto

O resultado principal do TG está congelado na tag `resultados-finais` (ADR 008) e não muda.
Dois pontos ficaram abertos:

1. em outubro/2023 o Transformer com ajuste fino errou menos que a LSTM escolhida (0,083 ×
   0,121, *post hoc*). A hipótese do texto é que a LSTM guarda o decaimento nos pesos,
   ajustado ao β de janeiro, e degrada mais quando β muda; não foi testada;
2. a LSTM com ajuste fino teve a melhor época entre 291 e 299, no teto de 300: o treino
   parou antes de convergir. Além disso, no α_c as redes com ajuste fino erram mais que a
   média dos rótulos (MSE na validação 0,0042 e 0,0012 × 0,00015), porque a perda na escala
   original é dominada por β (0,11 de 0,14).

O `test_closed` (26–28/jan/2024) já foi aberto e não é reaberto. Para avaliar de forma limpa
um modelo novo é preciso um bloco que ninguém viu.

## Decisão

Um adendo exploratório, com orçamento de um dia de máquina, em três partes. Todos os
resultados são relatados, inclusive os nulos. O adendo entra no texto como seção própria
e não substitui o resultado da etapa 8.

### Parte 1 — teste de β (só inferência, `make addendum-beta`)

- **Dados:** Hawkes simulado com histórico vazio, levado à grade de 100 ms como no
  pré-processamento (um evento por célula, `to_grid` do diagnóstico, sem deslocamento) e
  cortado nos primeiros 1.000 eventos, 400 trajetórias por β ∈ {1,2; 2,0; 2,7; 3,5; 4,5}. Fixos no centro dos rótulos de `calibration_train`: μ = 0,23,
  g = (α_s + α_c)/β = 0,50, α_c = 0,001; α_s = gβ − α_c. A seed de cada trajetória é
  `20261102 + 1000·i_β + j`.
- **Alvo:** o MLE de cada trajetória (5 inícios, como nos dados reais), porque é o que as
  redes aprenderam a imitar. Secundário: o θ verdadeiro.
- **Redes:** os pesos congelados de `lstm_lee/finetune` e `transformer/finetune` (ensemble
  das 5 seeds), sem retreino.
- **Hipótese H-β:** a perda da LSTM cresce mais que a do Transformer quando β se afasta de
  2,7 para cima. Estatística: Δ = d̄(4,5) − d̄(2,7), com d = perda_LSTM − perda_Transformer
  por trajetória (perda = erro quadrático médio nos 4 parâmetros). H-β é sustentada se o IC
  bootstrap 95% de Δ (2.000 reamostragens, independentes nos dois β) ficar acima de zero.
  β ∈ {1,2; 2,0; 3,5} são descritivos.

### Parte 2 — receita de ajuste fino (`make addendum-train`)

Mesmos dados (`calibration_train` para treino, `validation` para seleção), mesmas 5 seeds,
mesmo ponto de partida (`best.pt` sintético), lr 10⁻⁴ e lote do ADR 008. Braços novos:

| Braço | Mudança em relação a `finetune` | Redes |
|---|---|---|
| `finetune_long` | teto de 1.000 épocas (paciência 30 mantida) | LSTM |
| `finetune_sched` | teto 1.000; aquecimento linear em 5 épocas; lr × 0,5 após 10 épocas sem melhora (mínimo lr/100); corte da norma do gradiente em 1,0 | LSTM, Transformer |
| `finetune_std` | `finetune_sched` + perda padronizada: Σ_p MSE_p / max(Var_p, 0,01), Var_p a variância dos rótulos de treino | LSTM, Transformer |

Em todos os braços, o checkpoint, a parada antecipada e a redução no platô seguem o MSE
de validação na escala original, como no ADR 008; a padronização muda só a perda de treino.

O Transformer não recebe `finetune_long`: suas 5 seeds pararam por paciência entre as
épocas 53 e 97, longe do teto, de modo que o braço seria idêntico ao original. O cosseno
proposto no guia foi trocado pela redução no platô porque, com parada antecipada, o
horizonte do cosseno não é conhecido.

**Seleção** (igual à do ADR 008): menor MSE macro do ensemble na validação, na escala
original, entre os cinco braços novos. Fica registrada em
`artifacts/addendum_009/summary.json` antes de a semana nova ser lida.

### Parte 3 — semana nova (`make addendum-holdout`)

- **Bloco:** BTCUSDT e ETHUSDT, `um_futures`, `bookTicker`, 5–11/fev/2024 (segunda a
  domingo), processado como o período primário (grade de 100 ms, mesmos *tick sizes*) e
  cortado em janelas de 1.000 eventos sem sobreposição, como outubro. Escolhido antes do
  download; nenhum dado desse período foi visto.
- **Rótulo:** MLE com 5 inícios; janelas sem convergência saem.
- **Comparação principal:** braço escolhido na validação × `lstm_lee/finetune` congelado,
  pareada por janela (diferença média com IC bootstrap 95%, Wilcoxon). Uma única
  comparação principal, sem correção.
- **Secundárias**, com correção de Holm entre elas: cada um dos outros quatro braços ×
  `lstm_lee/finetune`; LSTM × Transformer dentro de `finetune_sched` e de `finetune_std`.
- **Também relatados:** MSE e R² por parâmetro, a média dos rótulos de treino como
  referência e o `transformer/finetune` congelado.
- O bloco é lido uma única vez, com `--open-holdout`.

Outubro/2023 é avaliado com todos os braços, como análise exploratória (já foi visto).

## Consequências

O `test_closed` continua sendo lido só pela etapa 8. Se o braço novo vencer na semana nova,
o texto diz que a receita de treino melhora o escopo A; o modelo escolhido no TG continua
sendo o da etapa 4. Teto do adendo: 12 h de máquina.

## Emenda (2026-11-02, durante o treino, antes de abrir a semana nova)

Entra um sexto candidato na seleção, motivado pelo teste de β (a LSTM erra menos dentro e
abaixo da faixa de janeiro, o Transformer acima dela): `mix/lstm+transformer`, a média
simples das previsões do melhor LSTM e do melhor Transformer na validação, cada um escolhido
entre o seu `finetune` congelado e os seus braços novos. A regra de seleção não muda (menor
MSE macro do ensemble na validação, agora entre seis candidatos). Na semana nova, a mistura
entra nas comparações secundárias contra `lstm_lee/finetune` (com Holm) se não for a
escolhida. Nenhum dado da semana nova foi rotulado ou avaliado até aqui.

## Desvio de processamento (antes de ler a semana nova)

Os arquivos `bookTicker` de fevereiro/2024 vêm com as linhas intercaladas fora da ordem do
tempo (no de 05/02 do BTCUSDT, metade das linhas de cada bloco volta no tempo); as cotações
são válidas (nenhuma inválida ou cruzada). `scripts/process_holdout_009.py` ordena cada
arquivo por `transaction_time` e `update_id`, confere que as duas chaves concordam e então
aplica o mesmo processamento do período primário. Nenhum rótulo nem métrica foi calculado
nesse passo.

## Resultado da parte 1 (teste de β)

400 trajetórias por β, MLE convergiu em todas; 1,4 min. Perda média (erro quadrático médio
nos 4 parâmetros contra o MLE), ensemble das 5 seeds:

| β verdadeiro | 1,2 | 2,0 | 2,7 | 3,5 | 4,5 | 6,0* | 8,0* |
|---|---|---|---|---|---|---|---|
| β mediano do MLE na grade | 0,93 | 1,45 | 1,86 | 2,28 | 2,73 | 3,33 | 3,96 |
| LSTM com ajuste fino | 0,017 | 0,007 | 0,022 | 0,028 | 0,028 | 0,037 | 0,125 |
| Transformer com ajuste fino | 0,029 | 0,097 | 0,114 | 0,091 | 0,053 | 0,038 | 0,059 |

\* Extensão fora do protocolo, acrescentada depois de ver a grade registrada
(`beta_sweep_extension.json`).

**Pela regra registrada, H-β é sustentada** (Δ = 0,067, IC 95% [0,061; 0,073]), mas a
premissa da regra falhou: a grade de 100 ms funde eventos próximos e puxa o β do MLE para
baixo, de modo que β verdadeiro 4,5 dá rótulo 2,73, o centro de janeiro, e não um
afastamento. Lida no espaço do rótulo, a evidência é outra: a LSTM erra menos que o
Transformer dentro e abaixo da faixa de janeiro (onde o Transformer é que degrada), empata
perto do β de outubro (3,19) e perde acima da faixa de treino (95% dos rótulos abaixo de
3,74), onde o seu erro quadruplica. A hipótese do texto vale só na direção de β acima de
janeiro, e o β sozinho não explica a diferença de outubro (que também tem μ e α_s maiores).
Achado lateral: o β que o MLE estima na grade fica entre 50% e 80% do β em tempo contínuo nessa
faixa.

## Seleção (parte 2, antes de abrir a semana nova)

25 treinos concluídos (fila com 9 workers em CPU e 1 em MPS). MSE macro do ensemble das 5
seeds na validação, escala original:

| Candidato | MSE na validação | Melhores épocas |
|---|---|---|
| `lstm_lee/finetune` (congelado) | 0,0359 | 291–299 |
| `transformer/finetune` (congelado) | 0,0413 | 53–97 |
| `lstm_lee/finetune_long` | **0,0294** | 158, 308, 441, 836, 925 |
| `lstm_lee/finetune_sched` | 0,0358 | 183–468 |
| `transformer/finetune_sched` | 0,0448 | 23–52 |
| `lstm_lee/finetune_std` | 0,0354 | 204–521 |
| `transformer/finetune_std` | 0,0431 | 23–59 |
| `mix/lstm+transformer` (`finetune_long` + `transformer/finetune`) | 0,0304 | — |

**Escolhido: `lstm_lee/finetune_long`.** `summary.json` sha256
`60c6567057d1f8e7f01c0fa1438a64f2ce5d30f49a9058cfa01c6b75fb818f92`, gravado com
`holdout_opened: false`; este commit precede a leitura da semana nova.

## Resultado da parte 3 (semana nova, lida uma vez)

711 janelas (5–11/fev/2024). MSE macro do ensemble; outubro é exploratório.

| Candidato | Semana nova | Outubro |
|---|---|---|
| média dos rótulos de treino | 0,1826 | 0,629 |
| `lstm_lee/finetune` (congelado) | 0,0264 | 0,121 |
| `transformer/finetune` (congelado) | 0,0285 | 0,083 |
| `lstm_lee/finetune_long` (escolhido) | **0,0212** | 0,116 |
| `lstm_lee/finetune_sched` | 0,0260 | 0,131 |
| `transformer/finetune_sched` | 0,0295 | 0,094 |
| `lstm_lee/finetune_std` | 0,0315 | 0,102 |
| `transformer/finetune_std` | 0,0330 | 0,085 |
| `mix/lstm+transformer` | **0,0204** | 0,087 |

**Comparação principal: sustentada.** `finetune_long` − `finetune` = −0,0052 (IC 95%
[−0,0060; −0,0044], Wilcoxon p < 10⁻³⁸), menor perda em 73% das janelas: o treino
original da LSTM parou cedo demais. **Secundárias (Holm):** a mistura também vence a LSTM
congelada (−0,0059, p < 10⁻¹²); as perdas padronizadas perdem (+0,005 e +0,007); o
cronograma empata. Nenhum braço Transformer vence a LSTM congelada na semana nova. A perda
padronizada levou o R² de α_c de −21 para −1,0 (LSTM) e 0,05 (Transformer), mas piorou β.
Em outubro os Transformers seguem melhores. Leitura: dentro do regime de treino a LSTM é a
melhor rede isolada; sob mudança de regime (outubro) o Transformer erra menos; a média das duas tem
a menor perda na semana nova e fica perto da melhor em outubro. O modelo escolhido no TG continua o da etapa 4.

## Viés da grade (*post hoc*, `scripts/run_grid_bias.py`)

Malha μ ∈ {0,05; 0,1; 0,23; 0,5; 1; 2} × β ∈ {1,2; …; 13}, g = 0,5, α_c = 0,001, 100
trajetórias por célula, levadas à grade e cortadas em 1.000 eventos; 107 s. A mediana do
MLE na grade define um mapa contínuo → rótulo, invertido por janela (inferência indireta).
O viés cresce com β e com μ; g também cai (0,5 → 0,22–0,48); μ quase não muda. Janeiro: β
mediano 2,71 na grade ↔ 4,7 em tempo contínuo (5–95%: 1,9–9,2), razão mediana 58%; 95% das
janelas dentro do mapa. Validação: 4,9 e 57%. Outubro: 3,19 ↔ 6,5. A frase do diagnóstico
de que o *jitter* quase não muda o β (2,71 → 2,77) continua certa, mas é incompleta: o
*jitter* desfaz empates, não os eventos fundidos. A inversão supõe g = 0,5 e α_c ≈ 0 e dá a
ordem de grandeza, não um rótulo corrigido.
