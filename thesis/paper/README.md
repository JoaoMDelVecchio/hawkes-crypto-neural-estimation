# Article version of the TG

`main.tex` is the journal-style article version of the thesis in `thesis/tg/`
(Elsevier `elsarticle`, preprint, author–year citations with `elsarticle-harv`).
Every number in it comes from the thesis chapters, the generated fragments in
`thesis/latex/` and the decision records `docs/decisions/008` and `009`.

- Bibliography: the shared `thesis/latex/references.bib`.
- Figures: read directly from `thesis/build/figures/` through `\graphicspath`
  (`posthoc_beta_verdadeiro_vs_estimado_en.pdf`, `real_data_residual_diagnostics.pdf`,
  `posthoc_calibracao_vs_prior_en.pdf`, `hawkes_volatility.pdf`); nothing is copied.
- Open items are marked `% TODO` in the source (authorship order, repository URL, and the
  addendum subsection on the new fine-tuning recipe and the 5–11 Feb 2024 week).

## Build

Requires TinyTeX (or TeX Live) with `elsarticle`
(`~/Library/TinyTeX/bin/universal-darwin/tlmgr install elsarticle` if missing).
Run from this folder; the PDF is written to `thesis/build/paper/main.pdf`.

```bash
cd thesis/paper
B=~/Library/TinyTeX/bin/universal-darwin
mkdir -p ../build/paper
$B/pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/paper main.tex
(cd ../build/paper && BIBINPUTS=../../latex: $B/bibtex main)
$B/pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/paper main.tex
$B/pdflatex -interaction=nonstopmode -halt-on-error -output-directory=../build/paper main.tex
```
