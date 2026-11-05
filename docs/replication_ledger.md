# Registro de reprodução — Lee (2023)

Paper: Kyungsub Lee, *Recurrent neural network based parameter estimation of Hawkes
model on high-frequency financial data*, Finance Research Letters 55 (2023), 103922.

## Especificações publicadas

| Item | Artigo |
|---|---|
| Processo | Hawkes marcado 2D, kernel exponencial e matriz simétrica |
| Parâmetros | `μ`, `α1` (self), `α2` (cross), `β` |
| Entrada | interchegadas + tipo de movimento (alta=1, baixa=2) |
| Filtro | mid-price observado a cada 0,1 s |
| Rede | duas camadas LSTM com 12 unidades e camada densa de 4 saídas |
| Treino | 75.000 amostras, Adam, MSE, geralmente mais de 300 épocas |
| Teste | 15.000 amostras |
| Robustez | softplus; prever `β - α1 - α2` |
| Referência reportada | MSE NN 0,0513; MLE 0,0417 |
| Tempo reportado | NN 0,0120 s; MLE 1,763 s |

## Ambiguidades e limitações

| Questão | Tratamento planejado |
|---|---|
| O artigo apresenta descrições conflitantes do comprimento das trajetórias | adotar 1.000 eventos como comprimento normativo e declarar a reconstrução |
| Faixas exatas dos parâmetros não estão publicadas | calibrar em cripto e manter experimento adicional com ponto fixo publicado |
| Dados NBBO originais não estão públicos | não alegar reprodução empírica exata; chamar de adaptação |
| Inicialização do MLE favorecida pelo parâmetro verdadeiro | reportar inicialização oracle e múltiplas inicializações realistas |
| Seeds, batch size e detalhes de treino ausentes | declarar escolhas antes do teste e executar sensibilidade |
| Definição operacional completa das janelas sobrepostas ausente | documentar stride e impedir vazamento entre splits |

## Níveis de evidência

- **R1 — reprodução direta:** detalhe explicitamente publicado.
- **R2 — reconstrução:** decisão necessária para preencher lacuna.
- **A1 — adaptação:** mudança exigida pelo mercado de cripto.
- **E1 — extensão:** Transformer/atenção.

Toda tabela final deve marcar o nível correspondente.

## Execuções iniciais

- O ponto fixo publicado foi simulado e o MLE convergiu em uma trajetória de 300 eventos.
- Uma execução exploratória com 1.024 trajetórias de 128 eventos e três épocas produziu
  MSE de teste 0,0448 para a LSTM e 0,0328 para o Transformer. Esse resultado é apenas
  um smoke test: não reproduz o tamanho, comprimento ou orçamento do paper.
- Na janela empírica primária, 12 ajustes MLE representativos foram executados,
  cobrindo símbolos, comprimentos e splits.
- Uma execução exploratória com as 12 janelas de 1.000 eventos rotuladas pelo MLE
  treinou ambos os modelos em dados reais: MSE de treinamento 0,5038 para LSTM e
  0,1800 para Transformer, com 100% das saídas válidas. Isso não é ainda uma
  avaliação fora da amostra; serve para validar a integração da pipeline.

## Pipeline final implementado

Em 1 de setembro de 2026, o protocolo escalonado foi materializado nos arquivos
`configs/local_smoke.yaml` e `configs/lee_synthetic_final.yaml`. A integração local
executou geração persistida, treino com checkpoints, teste e MLE justo no comprimento
final de 1.000 eventos. Foram usados somente 48/16/16 caminhos e duas épocas;
portanto, os números abaixo validam software, não hipóteses:

| Comprimento | LSTM MSE | Transformer MSE | MLE MSE | Validade NN |
|---|---:|---:|---:|---:|
| 1.000 | 0,2286 | 0,1176 | 0,0064 | 100% |

O MLE foi medido em apenas duas trajetórias, com inicialização pela taxa observada e
dois starts. Também foram validados: bloqueio explícito do teste empírico, treino
exclusivo em `calibration_train`, seleção em `validation` e inferência de robustez em
outubro com checkpoint congelado. Nenhum desses smoke tests autoriza declarar
superioridade do Transformer.

Em 28 de setembro de 2026, a execução científica final consolidou 75.000 trajetórias
de treino, 10.000 de validação e 15.000 de teste, todas com 1.000 eventos. LSTM e
Transformer foram agregados sobre cinco seeds independentes; o MLE foi ajustado
separadamente em cada trajetória de teste, com cinco starts heurísticos. O resumo
consolidado alimenta diretamente as tabelas e figuras da dissertação.

## Análise complementar (30 de outubro de 2026)

Detalhes no [ADR 007](decisions/007-analise-complementar.md).

- **E2 executado** (`make fixed-point`): 2.000 trajetórias de 1.000 eventos no ponto
  fixo. O MLE fica quase não viesado (0,307; 0,396; 0,708; 1,516), com desvios-padrão
  cerca de √2 vezes os de Lee, o esperado para metade do comprimento. As redes repetem
  o encolhimento de α_self e α_cross observado no artigo. MLE heurístico e oráculo
  produzem as mesmas estimativas.
- **Reconstruções R2 adicionais:** entrada `log1p(Δt)` e direção ±1 (Lee usa
  interchegadas e tipos 1/2); o artigo chama de "três camadas" as duas LSTM mais a densa.
- **Custo:** com gradiente analítico, o MLE custa ~10 ms por trajetória em um núcleo; a
  razão de ~150× do artigo (0,012 s contra 1,763 s) não se repete em lote unitário.
- **Calibração:** nenhuma das 1.063 janelas de `calibration_train` cai dentro das faixas
  do protocolo sintético; a adaptação empírica (A1) exige prior recalibrado.

## Etapa empírica e congelamento (2 de novembro de 2026)

Detalhes no [ADR 008](decisions/008-escopo-empirico-e-idioma.md).

- **Diagnóstico** (`make diagnostics`): Hawkes ≫ Poisson (LR rejeita em 99,8% das janelas),
  mas sobra dependência nos resíduos (Ljung–Box 47% contra 5% no controle na mesma grade).
- **Escopo A** (`make empirical-models`, 3,0 h): redes aprendem o rótulo do MLE em janelas
  reais. Escolhida na validação: LSTM com ajuste fino a partir dos pesos sintéticos.
- **Teste aberto uma vez** (`make empirical-test`): MSE 0,044 (LSTM escolhida), 0,040
  (Transformer com ajuste fino), 0,218 (média dos rótulos).
- **Volatilidade C4** (`make volatility`): nível certo (razão ≈ 1,27 para o MLE) e boa
  ordenação com o MLE (Spearman 0,78 no teste); a rede ordena pior (0,28).
- **Outubro/2023** (`make robustness-scope-a`): o Transformer com ajuste fino degrada menos.
- **Congelamento:** tag `resultados-finais` e release privada com os artefatos completos.
- **Depois da tag:** o diagnóstico da etapa 3 foi repetido nas 274 janelas do teste
  (`run_real_data_diagnostics.py --split test_closed --open-test`): Hawkes ≫ Poisson em 100%,
  Ljung–Box 47% contra 4% no controle. O JSON foi anexado à release `resultados-finais`.

## Adendo depois do congelamento (ADR 009)

Detalhes no [ADR 009](decisions/009-adendo-ajuste-fino-e-beta.md). Não muda o modelo escolhido.

- **Teste de β** (`make addendum-beta`): a grade de 100 ms puxa o β do MLE para baixo; a LSTM
  erra menos dentro e abaixo da faixa de janeiro e o Transformer acima dela.
- **Viés da grade** (`scripts/run_grid_bias.py`, *post hoc*): o rótulo é ~58% do β contínuo
  (janeiro 2,71 ↔ 4,7; outubro 3,19 ↔ 6,5); o *jitter* não desfaz isso.
- **Receita de ajuste fino** (`make addendum-train`, `make addendum-select`): 25 treinos;
  escolhida na validação a LSTM com teto de 1.000 épocas (0,0294 × 0,0359).
- **Semana nova** (`make addendum-holdout`, 5–11/fev/2024, lida uma vez): 0,0212 × 0,0264 da
  LSTM congelada (IC 95% da diferença [−0,0060; −0,0044]); média LSTM+Transformer 0,0204.
- **Escopo B reduzido** (`make scope-b`, ADR 010, ~3 h): prior a partir dos rótulos reais
  invertidos para tempo contínuo, simulado na grade; LSTM pré-treinada e com ajuste fino.
  Segunda semana nova (4–10/mar/2024, lida uma vez): 0,040 × 0,109 da LSTM de 1.000 épocas
  (IC 95% [−0,082; −0,056]); fevereiro 0,0153, outubro 0,034.
- **Volatilidade com as redes do adendo** (`make volatility-addendum`, ADR 011 parte 2):
  Spearman em fevereiro 0,63 da média LSTM+Transformer × 0,47 da LSTM congelada
  (IC [0,130; 0,181]); Transformer 0,69; MLE 0,77.
- **Correção da grade pela rede** (`make grid-correction`, ADR 011 parte 1): LSTM treinada com
  o θ contínuo verdadeiro; em 2.000 trajetórias novas, MSE macro 0,33 × 0,65 do MLE corrigido
  pelo mapa × 3,95 do MLE na grade (IC da principal [−0,405; −0,256]). Nas janelas reais a
  rede e o mapa discordam (β mediano 7,7 × 4,9 na validação).
- **Custo do MLE por implementação** (`scripts/benchmark_mle_variants.py`, ~3 min, um núcleo):
  10 trajetórias no ponto fixo de Lee, início no θ verdadeiro. Em 2.000 eventos: recursão com
  gradiente analítico 1,7 ms; soma dupla vetorizada 711 ms (gradiente numérico) e 3.358 ms
  (Nelder–Mead), contra 1.763 ms de Lee. Todas as variantes chegam à mesma estimativa
  (diferença máxima 0,002). Saída em `artifacts/benchmarks/mle_variants.json`.
