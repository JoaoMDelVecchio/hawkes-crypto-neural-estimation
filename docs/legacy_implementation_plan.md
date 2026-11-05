# Plano de Implementação — TG-2

**Projeto:** Hawkes Process Estimation on High-Frequency Cryptocurrency Data via Neural Networks
**Autor:** João Matheus Del Vecchio · **Orientador:** Prof. Dr. Mauri Aparecido de Oliveira (ITA)
**Base:** proposta aceita como TG-1 (`Report_TG1.pdf`) · **Período de execução:** segunda metade de 2026 (TG-2)

Este documento traduz a proposta de TG-1 em um plano de execução: fases, tarefas concretas,
entregáveis, critérios de aceite e dependências. Ele assume o cronograma indicativo da
Figura 7.1 do relatório (ago–dez/2026) e o detalha em nível de implementação.

---

## 1. Escopo herdado da proposta (não renegociável sem novo gate com o orientador)

- Modelo: processo de Hawkes 2D simétrico marcado, kernel exponencial, em regime de alta
  frequência (mid-price filtrado a 0,1 s) — não ultra-alta frequência.
- Dados: BTCUSDT e ETHUSDT (perpétuos USDT-margined), via Binance Public Data
  (bookTicker + aggTrades).
- Fora de escopo: modelagem completa do livro de ofertas, previsão de direção de preço,
  execução de ordens.
- Entregável mínimo (C1+C2): réplica da LSTM de Lee (2023) com MSE da mesma ordem do MLE
  e tempo de inferência < 1% do MLE, rodando em dados reais de cripto.
- Entregáveis de maior contribuição (C3+C4): arquitetura com atenção igual/melhor que a
  LSTM em simulação; série de volatilidade de Hawkes intradiária estável e consistente com
  realized volatility.

## 2. Critérios de sucesso (referência cruzada com as fases abaixo)

| Critério | Descrição | Fase que entrega | Prioridade |
|---|---|---|---|
| C1 | Réplica de Lee (2023): MSE ~ MLE, inferência < 1% do MLE | Fase 4 | Mínimo |
| C2 | Pipeline ponta a ponta em dados reais, respeitando positividade e g < 1 | Fases 1–4 | Mínimo |
| C3 | Ao menos uma arquitetura com atenção com MSE ≤ LSTM baseline (simulação) | Fase 5 | Incremento |
| C4 | Volatilidade intradiária estável e consistente com realized volatility | Fase 7 | Incremento |

## 3. Fases de implementação

### Fase 1 — Coleta e limpeza de dados de alta frequência

- **Objetivo:** produzir séries de eventos limpas (tempos, tipo up/down, marcas em ticks)
  para BTCUSDT e ETHUSDT a partir do Binance Public Data.
- **Tarefas:**
  1. Implementar `src/data/download_binance.py`: download dos dumps diários (.zip → CSV)
     de `bookTicker` e `aggTrades`, com checagem de integridade e retomada de download.
  2. Implementar `src/data/clean.py`: reconstrução do mid-price m(t) = (bid+ask)/2,
     amostragem na grade de 0,1 s, detecção de mudanças de preço (up/down), extração de
     tempos de inter-chegada e marcas (em unidades de tick, via `PRICE_FILTER` do
     `exchangeInfo`).
  3. Validar a série de eventos contra `aggTrades` (contagem de negócios vs. eventos de
     preço) como checagem de sanidade.
  4. Decidir e documentar a janela de datas definitiva (a proposta deixa isso
     parametrizável) e se um período de estresse de alta volatilidade será incluído.
  5. Avaliar a necessidade de reclocking em tempo de negócio (business time) dado que o
     mercado cripto opera 24/7 (mitigação do risco de não-estacionariedade, ver §5).
- **Entregável:** `data/processed/{symbol}_{window}.parquet` com colunas
  `timestamp, type (up/down), inter_arrival, jump_size_ticks`; notebook de validação em
  `notebooks/01_data_validation.ipynb`.
- **Critério de aceite:** série de eventos sem gaps não documentados, jump sizes
  consistentes com o tick size do contrato, contagem de eventos compatível com aggTrades
  (diferença justificada e documentada).
- **Dependências:** nenhuma. **Bloqueia:** Fase 3 (calibração de faixas de parâmetro),
  Fase 4/6 (avaliação em dados reais).

### Fase 2 — Motor de simulação Hawkes (Ogata thinning)

- **Objetivo:** gerador de trajetórias sintéticas com parâmetro conhecido, para treinar e
  validar os estimadores neurais.
- **Tarefas:**
  1. Implementar `src/simulate/hawkes_sim.py`: thinning de Ogata para o modelo marcado
     2D (Eq. 2.4 do relatório), com verificação de que a razão de ramificação simulada
     bate com g = (α₁+α₂)/β.
  2. Gerador de dataset de treino: milhares de trajetórias de ~2000 passos, parâmetros
     sorteados em faixas que serão calibradas por MLE em janelas reais (depende da Fase 3
     para o range final, mas pode iniciar com faixas da literatura/Lee 2023).
  3. Testes unitários: recuperação de parâmetro em regime não-crítico e próximo do
     crítico (g → 1).
- **Entregável:** `scripts/generate_training_set.py` produzindo `data/processed/synthetic_train.parquet`
  (ou formato equivalente) + testes em `tests/test_hawkes_sim.py`.
- **Critério de aceite:** distribuição empírica da razão de ramificação nas trajetórias
  simuladas bate com o valor teórico dentro de tolerância; testes passando.
- **Dependências:** nenhuma dura (pode rodar em paralelo à Fase 1). **Bloqueia:** Fase 4
  (dataset de treino da LSTM).

### Fase 3 — Baseline MLE e replicação do protocolo de Lee (2023)

- **Objetivo:** estimador de referência para comparação de acurácia e para calibrar as
  faixas de parâmetro do simulador.
- **Tarefas:**
  1. Implementar `src/estimate/mle.py`: log-verossimilhança vetorizada (Eq. 2.5) e
     otimização numérica (`scipy.optimize`, L-BFGS-B) sobre {μ, α₁, α₂, β}, com múltiplos
     pontos iniciais.
  2. Validar o MLE em trajetórias simuladas com parâmetro conhecido (erro de recuperação
     aceitável).
  3. Rodar o MLE em janelas reais (Fase 1) para calibrar as faixas de parâmetro usadas na
     Fase 2 e obter a métrica de referência para C1.
- **Entregável:** `src/estimate/mle.py` + relatório de calibração
  (`notebooks/02_mle_calibration.ipynb`) com faixas de parâmetro por símbolo/janela.
- **Critério de aceite:** erro de recuperação de parâmetro em simulação dentro do esperado
  pela literatura; tempo de execução do MLE registrado (referência para o ganho de custo
  computacional da rede).
- **Dependências:** Fase 1 (para calibração em dados reais) e Fase 2 (para validação em
  simulação) — pode começar a implementação do otimizador sem esperar as duas.

### Fase 4 — LSTM de referência (réplica de Lee 2023) e validação

- **Objetivo:** entregável mínimo do TG-2 (C1+C2).
- **Tarefas:**
  1. Implementar `src/models/lstm_baseline.py`: LSTM de 3 camadas, camada de saída com 4
     unidades e ativação softplus prevendo β−α₁−α₂ (garante positividade e g<1).
  2. Loop de treino (Adam, perda MSE) sobre o dataset sintético da Fase 2.
  3. Validar em trajetórias simuladas held-out (MSE vs. MLE) e depois em dados reais
     (Fase 1), respeitando as restrições do modelo.
  4. Medir tempo de inferência e comparar contra o tempo do MLE (Fase 3) — meta: < 1%.
- **Entregável:** checkpoint treinado + relatório de validação
  (`notebooks/03_lstm_validation.ipynb`) cobrindo C1 e C2.
- **Critério de aceite:** MSE da mesma ordem de grandeza do MLE em simulação; pipeline
  roda ponta a ponta em dados reais sem violar positividade/g<1; tempo de inferência
  medido e comparado.
- **Dependências:** Fases 2 e 3. **Marco:** conclusão da Fase 4 fecha o entregável mínimo
  do TG-2 — ponto natural de checkpoint com o orientador antes de avançar às extensões.

### Fase 5 — Arquiteturas alternativas (GRU, LSTM profunda, atenção/Transformer)

- **Objetivo:** investigar se arquiteturas alternativas igualam ou superam a LSTM de
  referência (C3), com foco no encoder de atenção.
- **Tarefas:**
  1. Implementar `src/models/gru_deep.py` (GRU e LSTM profunda) com a mesma interface de
     entrada/saída da baseline, para comparação justa.
  2. Implementar `src/models/attention_encoder.py` (Transformer, Vaswani et al. 2017),
     tratando a instabilidade de treino como risco conhecido (começar simples, ajustar
     regularização e hiperparâmetros).
  3. Treinar todas as variantes no mesmo dataset sintético e protocolo da Fase 4.
- **Entregável:** checkpoints das 3 variantes + registro de hiperparâmetros em `configs/`.
- **Critério de aceite:** todas as variantes treinam de forma estável (sem divergência);
  ao menos uma variante com atenção atinge MSE ≤ LSTM baseline em simulação.
- **Dependências:** Fase 4 (interface e protocolo de treino já validados).

### Fase 6 — Experimentos comparativos e métricas

- **Objetivo:** comparação sistemática de acurácia, custo computacional e robustez entre
  todas as arquiteturas.
- **Tarefas:**
  1. Implementar `src/eval/metrics.py`: MSE contra MLE (simulação e dados reais), tempo de
     inferência, e testes de robustez (sensibilidade a ruído de microestrutura, a
     diferentes regimes de mercado).
  2. Rodar a bateria completa para LSTM, GRU, LSTM profunda e Transformer, em BTCUSDT e
     ETHUSDT.
  3. Consolidar tabela comparativa (equivalente à Tabela 3.1 do relatório, mas com
     resultados próprios) para a monografia.
- **Entregável:** `notebooks/04_comparative_results.ipynb` + tabela/figuras finais.
- **Critério de aceite:** todas as arquiteturas avaliadas sob o mesmo protocolo; resultados
  reprodutíveis (seed fixa, config versionada).
- **Dependências:** Fase 5.

### Fase 7 — Aplicação: volatilidade de Hawkes intradiária

- **Objetivo:** C4 — série de volatilidade intradiária estável e consistente com o
  benchmark de realized volatility.
- **Tarefas:**
  1. Implementar `src/volatility/hawkes_volatility.py`: cálculo da forma fechada da
     volatilidade em função de {μ, α₁, α₂, β} (Eq. 2.3/2.4 do relatório) usando os
     parâmetros estimados pela(s) melhor(es) rede(s) da Fase 6.
  2. Calcular realized volatility (Andersen, Bollerslev e Labys, 2003) como benchmark.
  3. Comparar as duas séries (estabilidade, correlação, defasagens).
- **Entregável:** `notebooks/05_volatility_application.ipynb`.
- **Critério de aceite:** série de volatilidade de Hawkes sem instabilidades numéricas
  (g < 1 mantido) e correlação/consistência qualitativa com o benchmark, documentada.
- **Dependências:** Fase 6 (escolha do melhor estimador).

### Fase 8 — Análise de resultados e robustez

- **Objetivo:** consolidar achados, revisitar hipótese e objetivos específicos da
  proposta, identificar limitações.
- **Tarefas:** revisão cruzada dos resultados das Fases 6–7 contra a hipótese e os
  objetivos específicos (i)–(vii) do relatório; discussão de limitações (não-
  estacionariedade 24/7, qualidade dos dados, generalização entre símbolos).
- **Entregável:** seção de resultados/discussão rascunhada para a monografia.
- **Dependências:** Fases 6 e 7.

### Fase 9 — Escrita da monografia final

- **Tarefas:** expandir o relatório de TG-1 (introdução, fundamentação, trabalhos
  relacionados já escritos) com metodologia detalhada, resultados, discussão e conclusão;
  revisão de formatação ABNT/ITA.
- **Entregável:** monografia completa (draft para revisão do orientador, depois versão
  final).
- **Dependências:** Fase 8.

### Fase 10 — Preparação da defesa

- **Tarefas:** slides de defesa, ensaio de apresentação, lista de perguntas antecipadas
  (mesmo padrão do `Proposta_TG1/presentation.tex` usado na defesa do TG-1).
- **Dependências:** Fase 9 (ou versão avançada o suficiente da monografia).

## 4. Cronograma indicativo

Segue a distribuição da Figura 7.1 da proposta, com nomes de fase alinhados às seções
acima (ajustar datas exatas junto ao orientador; hoje é 31/ago/2026):

| Período | Fases em foco |
|---|---|
| Ago (semanas 1–4) | Fase 1 (dados) + início da Fase 2 (simulador) |
| Set (semanas 1–2) | Fase 2 (simulador) + Fase 3 (MLE/replicação) |
| Set (semanas 3–4) | Fase 4 (LSTM baseline & validação) |
| Out | Fase 5 (arquiteturas alternativas) + início da Fase 6 |
| Nov (semanas 1–2) | Fase 6 (experimentos comparativos) |
| Nov (semanas 3–4) | Fase 7 (volatilidade) + Fase 8 (análise/robustez) |
| Dez (semanas 1–3) | Fase 9 (escrita da monografia) |
| Dez (última semana) | Fase 10 (preparação da defesa) |

O núcleo (C1+C2, Fases 1–4) é priorizado nas primeiras semanas para garantir o
entregável mínimo antes de investir nas extensões (Fases 5–7).

## 5. Riscos e mitigação (herdados da proposta, com responsável de mitigação por fase)

| Risco | P × I | Mitigação | Fase(s) afetada(s) |
|---|---|---|---|
| Ruído de microestrutura e qualidade dos dados de cripto | M × H | Filtragem do mid-price a 0,1s; validação contra aggTrades | Fase 1 |
| Má especificação do modelo (efeito subdue, α negativo) | M × M | Saída softplus prevendo β−α₁−α₂; treino híbrido (simulação+MLE) | Fases 2–4 |
| Não-estacionariedade do mercado cripto (24/7, regimes) | H × M | Estimação por janela/segmento com sobreposição; análise por regime | Fases 1, 6, 8 |
| Custo/instabilidade do treino com atenção | M × M | Começar com modelos simples; regularização; LSTM garante entrega mínima | Fase 5 |
| Scope creep no cronograma do TG-2 | M × H | Priorizar C1+C2; arquiteturas de atenção e volatilidade como incremento | Todas |

## 6. Setup técnico inicial

Um scaffold de projeto já foi criado (ver arquivos entregues junto a este plano) com a
seguinte estrutura, espelhando as fases acima:

```
tg2_scaffold/
  README.md              # como rodar, estrutura, reprodutibilidade
  requirements.txt        # numpy, scipy, pandas, pyarrow, torch, matplotlib, tqdm,
                           # requests, pyyaml, pytest, scikit-learn, statsmodels
  .gitignore
  src/
    data/                 # Fase 1: download_binance.py, clean.py
    simulate/              # Fase 2: hawkes_sim.py
    estimate/               # Fase 3: mle.py
    models/                # Fases 4–5: lstm_baseline.py, gru_deep.py, attention_encoder.py
    eval/                   # Fase 6: metrics.py
    volatility/             # Fase 7: hawkes_volatility.py
  notebooks/               # notebooks de validação/resultados por fase (01–05)
  data/{raw,processed}/    # dumps brutos e séries limpas (não versionados)
  configs/                 # janelas de data, hiperparâmetros, seeds
  tests/                   # testes unitários (simulador, MLE, pipeline)
  scripts/                 # CLIs (ex.: download_binance_data.py)
```

Cada módulo já contém um docstring com o objetivo, a referência à equação/seção do
relatório de TG-1 e uma lista `TODO` com as tarefas da fase correspondente — ponto de
partida direto para a implementação.

**Próximo passo imediato:** rodar `pip install -r requirements.txt`, implementar
`src/data/download_binance.py` (Fase 1, tarefa 1) e baixar uma janela pequena de teste
(1 semana) de bookTicker para BTCUSDT, validando o pipeline de ponta a ponta em escala
reduzida antes de baixar a janela completa.
