# ADR 010 — Escopo B reduzido: pré-treino na região dos dados e na grade

Status: **aceita**; protocolo congelado antes de gerar os dados sintéticos e de baixar a
semana de teste
Data: 2026-11-02

## Contexto

O ADR 008 deixou o escopo B (prior recalibrado) fora do TG. O adendo do ADR 009 mostrou dois
fatos que o tornam mais barato e mais preciso de definir: a LSTM com ajuste fino estava
subtreinada, e a grade de 100 ms faz o MLE estimar cerca de 58% do β em tempo contínuo. O
pré-treino sintético original usou trajetórias em tempo contínuo, num prior que não cobre os
dados, com o θ verdadeiro como alvo; o ajuste fino precisou então mudar de região e de alvo.

Este é um experimento de segundo plano. **Só entra no texto do TG se a comparação principal
for favorável**; o resultado, qualquer que seja, fica registrado aqui.

## Decisão

### Dados sintéticos (`scripts/run_scope_b.py --generate`)

- **Prior** (mistura): com probabilidade 0,75, uma janela de `calibration_train` sorteada;
  (μ, β) contínuos obtidos invertendo o mapa do viés da grade (`grid_bias.json`), cada um
  multiplicado por exp(N(0; 0,2²)); g = g do rótulo × U(1,0; 1,25), limitado a [0,05; 0,95];
  fração cruzada α_c/(α_s+α_c) igual à do rótulo. Com probabilidade 0,25, um ponto largo: log μ
  ~ U[log 0,05; log 2], log β ~ U[log 1,2; log 13], g ~ U[0,2; 0,9], fração cruzada 0 com
  probabilidade 0,5 e U[0; 0,5] caso contrário. α_c tem piso 10⁻⁴.
- **Trajetórias:** simuladas em tempo contínuo, levadas à grade de 100 ms (um evento por
  célula, `to_grid`) e cortadas nos primeiros 1.000 eventos. 20.000 de treino e 2.000 de
  validação; seed `20261104`.
- **Alvo:** o MLE na grade de cada trajetória (5 inícios, como nos dados reais), não o θ
  verdadeiro: o pré-treino passa a imitar o mesmo estimador que o ajuste fino. Saem as
  trajetórias sem convergência e as com β̂ > 20 (fora de qualquer rótulo real).
- **Controle de cobertura** (relatado, não decide nada): fração dos rótulos de
  `calibration_train` dentro do intervalo 1%–99% dos rótulos sintéticos, por parâmetro.

### Redes

- Só a LSTM (`lstm_lee`), as 5 seeds da configuração, CPU.
- **Pré-treino** do zero nos dados sintéticos: lr 10⁻³, lote 64, teto de 150 épocas,
  paciência 15, perda MSE na escala original; checkpoint pelo MSE de validação sintética.
- **Ajuste fino** a partir do pré-treino, receita do vencedor do ADR 009
  (`finetune_long`): `calibration_train`, lr 10⁻⁴, teto de 1.000 épocas, paciência 30,
  checkpoint pelo MSE de `validation`.
- **Candidatos:** `B/zero` (pré-treino, sem ajuste fino), `B/finetune` e `mix/B+transformer`
  (média de `B/finetune` com o `transformer/finetune` congelado, como a mistura do ADR 009).
  As métricas de validação ficam em `artifacts/scope_b_010/summary.json` antes de a semana de
  teste ser lida.

### Semana de teste

- **Bloco:** BTCUSDT e ETHUSDT, `um_futures`, `bookTicker`, 4–10/mar/2024 (segunda a
  domingo), escolhido antes do download, processado como a semana do ADR 009 (ordenação por
  `transaction_time` e `update_id`), janelas de 1.000 eventos sem sobreposição, rótulo MLE.
  Lido uma única vez (`--open-holdout`).
- **Comparação principal:** `B/finetune` × `lstm_lee/finetune_long` (ADR 009), pareada por
  janela; favorável se o IC bootstrap 95% da diferença média ficar abaixo de zero.
- **Secundárias** (Holm): `mix/B+transformer` × `mix/lstm+transformer` (ADR 009);
  `B/zero` × média dos rótulos de treino; `B/finetune` × `lstm_lee/finetune` (congelado).
- **Exploratório:** fevereiro/2024 (já lido no ADR 009) e outubro/2023.

## Consequências

Se a principal for favorável, proponho ao aluno uma subseção curta no adendo do TG e no
artigo; caso contrário, o experimento fica só neste registro. O modelo escolhido no TG não
muda em nenhum caso. Teto: 8 h de máquina.

## Resultado

**Execução.** 22.000 trajetórias geradas (nenhuma descartada; os rótulos reais caem dentro do
intervalo 1–99% do sintético em 98–99% das janelas, por parâmetro). Pré-treino: 5 seeds,
~2 h 7 min cada em paralelo, melhor época 145–150 (o teto de 150 foi atingido; MSE de
validação sintética 0,011–0,028). Ajuste fino: melhor época 291–648.

**Validação** (MSE macro do ensemble, antes de ler a semana de março; `summary.json`):

| Candidato | Validação |
|---|---|
| `B/zero` | 0,144 |
| `B/finetune` | **0,0202** |
| `mix/B+transformer` | 0,0196 |
| referência: `lstm_lee/finetune_long` (ADR 009) | 0,0294 |

**Semana de março** (1.860 janelas, 4–10/mar/2024, lida uma vez; `holdout.json` sha256
`5e1d144c…`). MSE macro do ensemble; fevereiro e outubro são exploratórios.

| Variante | Março (1.860) | Fevereiro (711) | Outubro (1.281) |
|---|---|---|---|
| média dos rótulos | 0,582 | 0,183 | 0,629 |
| `lstm_lee/finetune` (congelado) | 0,117 | 0,0264 | 0,121 |
| `lstm_lee/finetune_long` (ADR 009) | 0,109 | 0,0212 | 0,116 |
| `transformer/finetune` | 0,112 | 0,0285 | 0,083 |
| `mix/lstm+transformer` (ADR 009) | 0,102 | 0,0204 | 0,087 |
| `B/zero` | 0,122 | 0,212 | 0,141 |
| **`B/finetune`** | **0,040** | 0,0153 | **0,034** |
| `mix/B+transformer` | 0,056 | **0,0137** | 0,041 |

**Comparação principal: favorável.** `B/finetune` − `finetune_long` = −0,068 (IC 95%
[−0,082; −0,056]); a rede do escopo B erra menos em 72% das janelas. Mesmo sinal em fevereiro
(−0,006, IC [−0,008; −0,004]) e outubro (IC [−0,096; −0,069]).

**Secundárias (Holm), todas favoráveis:** `mix/B+transformer` × `mix/lstm+transformer`
−0,046 (IC [−0,053; −0,039]); `B/zero` × média dos rótulos −0,460; `B/finetune` × LSTM
congelada −0,077 (IC [−0,090; −0,064]). Todos os p de Holm < 10⁻¹⁰.

**Onde está o ganho.** Em março e outubro o erro das redes antigas vem sobretudo de μ (MSE
0,31 e 0,18 na `finetune_long`, R² 0,61 e 0,65); a rede do escopo B leva μ a 0,086 e 0,044
(R² 0,89 e 0,91), e β também melhora (0,097 → 0,064 em março). São as semanas mais distantes da
calibração (a média dos rótulos erra 3–3,5 vezes mais que em fevereiro). O prior largo do
pré-treino cobre essa região; a calibração de janeiro, sozinha, não. Sem ajuste fino o
escopo B não serve (`B/zero` é pior que as redes antigas em março e muito pior em fevereiro):
o simulador dá a forma, mas o nível só vem com o ajuste fino.

**Diferente do ADR 009,** aqui a mistura com o Transformer não ajuda em março nem em outubro:
a `B/finetune` sozinha é a melhor. O modelo escolhido no TG não muda (regra do ADR 008); o
resultado entra no adendo do texto como evidência de que um pré-treino sintético mais próximo
dos dados reais (o escopo B que o TG deixara para depois) reduz o erro fora da amostra mais
que qualquer variante anterior.
