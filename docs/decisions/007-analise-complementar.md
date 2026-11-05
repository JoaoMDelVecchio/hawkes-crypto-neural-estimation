# ADR 007 — Análise complementar e reinterpretação dos resultados

Status: **executada**
Data: 2026-10-30

## Contexto

A comparação final de 1.000 eventos (ADR 005/006) foi resumida apenas pelo MSE macro.
Esse resumo levou a três leituras frágeis: superioridade da LSTM sobre o Transformer,
superioridade das redes sobre o MLE e inferência neural muito mais rápida que o MLE.
O experimento E2 (ponto fixo do artigo), previsto no protocolo, não havia sido
executado, e as faixas do prior sintético não tinham sido calibradas nos dados.

## Decisão

Executar análises que não retreinam modelos, não alteram o protocolo e não abrem o
teste empírico:

1. recalcular as previsões por trajetória a partir dos `best.pt` congelados;
2. reportar R² por parâmetro em relação à média a priori, erro absoluto mediano e
   comparação pareada por trajetória (IC *bootstrap* e Wilcoxon), além dos testes
   entre seeds (Welch e Mann–Whitney);
3. diagnosticar o MLE: concentração do erro, verossimilhança em θ̂ contra θ verdadeiro
   e reajuste com 20 inícios;
4. medir o custo em benchmark controlado, sem outros jobs;
5. executar o E2 com MLE heurístico e oráculo;
6. ajustar o MLE em todas as janelas de `calibration_train` (bloco autorizado para
   calibração) e comparar com o suporte do prior.

Comandos: `make fixed-point` e `make posthoc`.

## Evidência

- Reprodutibilidade: MSE recalculado igual ao reportado (diferença relativa ≤ 1,3e-8);
  o código do MLE reproduz `records.jsonl` exatamente.
- LSTM × Transformer: entre seeds, Welch p = 0,19; no pareado, o ensemble LSTM erra menos
  em 70% das trajetórias e em todos os regimes.
- Redes × MLE: 1% das trajetórias concentra 94% do erro quadrático de β do MLE (g baixo);
  em todas as 300 piores, ℓ(θ̂) ≥ ℓ(θ verdadeiro); 20 inícios não mudam o resultado.
  Na mediana, LSTM e MLE empatam; o MLE tem R² 0,90 em α_self e α_cross contra 0,82 da LSTM.
- E2: padrão de Lee reproduzido (encolhimento das redes, MLE quase não viesado); a
  inicialização-oráculo produz as mesmas estimativas que a heurística.
- Custo: MLE com 5 inícios custa 9,6 ms por trajetória em um núcleo (2,5 ms de parede com
  4 processos). O valor de 105 ms registrado no ADR 006 foi medido com o MLE concorrendo
  com o treino na GPU.
- Calibração: nenhuma das 1.063 janelas cai dentro de todas as faixas do prior;
  α_cross < 1e-3 em 89% e β mediano de 2,71.

## Consequências

- O texto passa a afirmar: H3 não suportada no sintético; H2 parcialmente suportada
  (mesmo erro; vantagem de custo só em lote); vantagem das redes no MSE agregado =
  robustez em β mal identificado + encolhimento para o prior.
- A etapa empírica exige recalibrar o prior ou treinar com rótulos MLE em janelas reais
  antes de abrir `test_closed`.
- Os números de custo do ADR 006 permanecem como registro histórico, mas não devem ser
  usados na comparação de latência.
