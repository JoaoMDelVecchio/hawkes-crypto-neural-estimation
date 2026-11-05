# ADR 008 — Escopo empírico, título e idioma do TG-2

Status: **aceita**; etapa 3 executada, etapa 4 em execução
Data: 2026-11-02

## Contexto

O orientador e o aluno precisavam fechar quatro pontos antes da etapa empírica (roadmap,
etapa 2): título, idioma, escopo empírico e datas. A análise complementar (ADR 007) mostrou
que nenhuma janela real cai no prior sintético usado no treino, de modo que as redes atuais
não servem, como estão, para estimar parâmetros em cripto.

## Decisão

1. **Título:** mantém o registrado no TG-1, *Hawkes Process Estimation on High-Frequency
   Cryptocurrency Data via Neural Networks*. Não há pedido de alteração (Norma de TG, art. 3º,
   §2º).
2. **Idioma:** inglês. O modelo do ITA permite português ou inglês; Resumo e Abstract são
   obrigatórios nos dois casos.
3. **Escopo empírico:** diagnóstico de ajuste do Hawkes nos dados reais e escopo A (redes
   treinadas em janelas reais com rótulos MLE), com teto de 30 h de máquina. O escopo B (prior
   recalibrado e retreino sintético) fica para o mestrado.
4. **Datas:** o quanto antes; o aluno registra as datas oficiais quando forem fixadas.

Em aberto: composição da banca (necessária no envio do texto) e se a volatilidade de Hawkes
contra *realized volatility* (critério C4) entra no teste ou vira trabalho futuro.

## Protocolo empírico (congelado antes de abrir o teste)

**Diagnóstico** (`make diagnostics`, só `calibration_train`, 15–22/jan/2024, 1.063 janelas
de 1.000 eventos): resíduos por *time-rescaling* por dimensão com KS contra Exp(1) e
Ljung–Box (10 defasagens); razão de verossimilhança e AIC contra Poisson homogêneo 2D;
sensibilidade à grade de 100 ms (deslocamento uniforme dentro da célula e novo MLE); controle
simulado com o próprio Hawkes ajustado, em tempo contínuo e levado à grade.

**Redes em janelas reais** (`make empirical-models`):

- rótulo de cada janela: MLE com 5 inícios heurísticos (`configs/lee_local_optimized.yaml`);
- treino em `calibration_train` (1.063 janelas), seleção em `validation` (476 janelas,
  23–25/jan); o `test_closed` (274 janelas, 26–28/jan) não é lido;
- variantes, cada uma com as 5 seeds do protocolo: LSTM e Transformer treinados do zero
  (Adam, lr 10⁻³) e a partir do `best.pt` sintético da mesma seed (lr 10⁻⁴); mesmo lote,
  300 épocas e paciência 30 do ADR 005; MSE na escala original;
- referências sem treino: redes sintéticas como estão (*zero-shot*) e média dos rótulos de
  treino;
- escolha: menor MSE macro do ensemble das 5 seeds na validação, registrada em
  `artifacts/empirical_models/scope_a/summary.json`.

**Teste** (etapa 8, uma única vez, `make empirical-test`): rotula o `test_closed` com o MLE e
avalia os pesos já gravados, sem retreinar; o script recusa se a variante escolhida mudar.
Todas as variantes são reportadas; a escolhida é o resultado principal. A janela de robustez
(23–29/out/2023) é avaliada sem retreino.

**Volatilidade (C4, entra no TG por decisão do aluno em 2026-11-02):** `make volatility`.
Volatilidade de Hawkes de longo prazo do modelo simétrico com marcas,
σ²_H = 2Λ[Var(J) + E[J]²/(1 − a_s + a_c)²] com Λ = μ/(1 − a_s − a_c), a_s = α_self/β,
a_c = α_cross/β e J = |retorno log| de cada evento (Bacry et al., 2013), multiplicada pela duração
da janela; benchmark: variância realizada do log do mid-price amostrado a cada Δ ∈ {0,1; 1; 5;
10; 30} s (Andersen et al., 2003). Métricas, fixadas antes do teste: correlação de Spearman
entre janelas e razão mediana σ_H/σ_RV, com Δ = 10 s como principal, para o MLE e para o
ensemble da variante escolhida; estabilidade como variação absoluta mediana do log entre
janelas consecutivas, comparada à da RV. Desenvolvido e calibrado só em `calibration_train` e
`validation` (MLE: Spearman 0,74/0,67 e razão 1,27/1,24 em Δ = 10 s); no teste, roda uma vez
na etapa 8 com `--open-test`.

## Resultado da etapa 3

Hawkes contra Poisson: a razão de verossimilhança rejeita o Poisson em 99,8% das janelas
(ΔAIC mediano −721). O Hawkes exponencial simétrico, porém, não é o modelo verdadeiro: o
Ljung–Box rejeita independência dos resíduos em 47% das janelas×dimensões, contra 5% no
controle simulado na mesma grade. A grade sozinha faz o KS rejeitar 48% no controle (dados
reais: 70%), e suavizá-la quase não muda o MLE (β mediano 2,71 → 2,77). Leitura: o Hawkes
explica a maior parte do agrupamento, mas sobra dependência que um único núcleo exponencial
não captura. Por isso, nos dados reais, o alvo das redes é o MLE desse modelo, não um
parâmetro verdadeiro.

## Resultado da etapa 4 (congelado antes de abrir o teste)

Treino do escopo A concluído em 3,0 h de máquina (teto: 30 h). MSE macro na validação,
ensemble das 5 seeds: média dos rótulos 0,210; redes sintéticas sem treino 0,555 (LSTM) e
0,558 (Transformer); do zero 0,139 (LSTM) e 0,100 (Transformer); ajuste fino 0,036 (LSTM) e
0,041 (Transformer). **Variante escolhida: `lstm_lee/finetune`.** Limitação: as melhores épocas
da LSTM com ajuste fino ficaram em 291–299, perto do teto de 300; o Transformer com ajuste fino
parou por paciência (melhores épocas 23–67). `summary.json` congelado com SHA-256 `6cdcdc02e1c26949…`.

Volatilidade em Δ = 10 s (calibração/validação): MLE razão 1,27/1,24 e Spearman 0,74/0,67; rede
escolhida razão 1,30/1,24 e Spearman 0,40/0,47. A rede acerta o nível e ordena pior as janelas.

## Resultado da etapa 8 (teste aberto uma vez, em 2026-11-02)

Rótulos do teste: 274 janelas (26–28/jan), MLE convergiu em todas. Pesos gravados, sem
retreino. MSE macro do ensemble no teste: média dos rótulos 0,218; LSTM com ajuste fino
(escolhida) 0,044; Transformer com ajuste fino 0,040; do zero 0,250 (LSTM) e 0,169
(Transformer); sintéticas sem treino 0,250 e 0,275. A escolha não muda: a seleção foi feita na
validação.

Volatilidade no teste, Δ = 10 s: MLE razão 1,27 e Spearman 0,78; rede escolhida razão 1,38 e
Spearman 0,28. Outubro/2023 (1.281 janelas, sem retreino): média dos rótulos 0,629; LSTM com
ajuste fino 0,121; Transformer com ajuste fino 0,083.

Análise post hoc, fora do protocolo (`scripts/compare_scope_a_finetune.py`): diferença pareada
por janela LSTM − Transformer (ajuste fino) no teste 0,005, IC bootstrap 95% [−0,002; 0,011],
Wilcoxon p = 0,39 (indistinguíveis); em outubro 0,038 [0,032; 0,045], p < 10⁻³², Transformer
melhor em 64% das janelas.

## Consequências

- O texto passa a ser escrito em inglês e no formato do modelo do ITA (roadmap, etapa 6).
- Custo de máquina medido do escopo A: segundos para os rótulos e poucas horas para o treino,
  dentro do teto de 30 h.
