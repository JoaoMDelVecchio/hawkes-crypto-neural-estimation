# Slides da defesa (rascunho, etapa 11)

Beamer 16:9, em inglês, 28 quadros (23 principais, agradecimento e 4 de backup). As figuras
vêm de `thesis/build/figures/` e os números, de `thesis/latex/empirical_results.tex`,
`thesis/latex/en/*.tex` e dos capítulos de `thesis/tg/`. Se esses arquivos forem
regenerados, os números dos slides precisam ser conferidos à mão: eles foram copiados, não
gerados.

Compilar a partir desta pasta (TinyTeX; precisa do pacote `beamer`):

```bash
mkdir -p ../build/slides && pdflatex -output-directory=../build/slides main.tex && pdflatex -output-directory=../build/slides main.tex
```

Saída: `thesis/build/slides/main.pdf`. Pendente (etapa 12): data da defesa na página de título.
