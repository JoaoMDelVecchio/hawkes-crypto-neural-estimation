# Documentação acadêmica em LaTeX

Esta pasta contém três documentos diferentes:

1. `reports/results_report.tex`: relatório técnico dos artefatos e resultados já
   obtidos.
2. `dissertation/dissertation_draft.tex`: rascunho da dissertação, com estrutura
   acadêmica completa. Seções ainda dependentes dos experimentos finais são marcadas.
3. `manual/step_by_step_manual.tex`: material didático para estudar e preparar a
   apresentação oral.

Os PDFs são gravados em `build/`. A configuração compartilhada e as referências ficam
em `latex/`.

Fragmentos gerados por script (não editar à mão):

| Fragmento | Gerado por |
|---|---|
| `latex/final_results.tex` | `scripts/generate_final_report_fragment.py` (`make local-final`) |
| `latex/e2_fixed_point.tex` | `scripts/run_fixed_point_e2.py` (`make fixed-point`) |
| `latex/posthoc_results.tex` | `scripts/generate_posthoc_fragment.py` (`make posthoc`) |

## Compilação

```bash
cd thesis
make all
```

Com TinyTeX instalado fora do `PATH`, o Makefile procura automaticamente
`~/Library/TinyTeX/bin/universal-darwin`.

## Regra de rigor

Valores de Lee (2023), resultados produzidos pelo projeto e metas futuras são
identificados separadamente. Os resultados sintéticos finais (1.000 eventos, cinco
seeds) estão congelados; diferenças entre estimadores só são afirmadas com o teste
estatístico correspondente (ADR 007). A etapa empírica ainda não foi avaliada.

