#!/usr/bin/env bash
# Builds the presentation. Requires XeLaTeX (TeX Live) + the Lato font.
set -e
for i in 1 2 3; do xelatex -interaction=nonstopmode presentation.tex; done
echo "OK -> presentation.pdf"
