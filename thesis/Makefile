TINYTEX_BIN := $(HOME)/Library/TinyTeX/bin/universal-darwin
PDFLATEX := $(if $(wildcard $(TINYTEX_BIN)/pdflatex),$(TINYTEX_BIN)/pdflatex,pdflatex)
BIBTEX := $(if $(wildcard $(TINYTEX_BIN)/bibtex),$(TINYTEX_BIN)/bibtex,bibtex)
OUT := build

DOCS := results_report dissertation_draft step_by_step_manual

.PHONY: all clean results dissertation manual

all: results dissertation manual

results:
	@mkdir -p $(OUT)/results
	cd reports && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/results results_report.tex
	cd $(OUT)/results && BIBINPUTS=../../latex: $(BIBTEX) results_report
	cd reports && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/results results_report.tex
	cd reports && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/results results_report.tex

dissertation:
	@mkdir -p $(OUT)/dissertation
	cd dissertation && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/dissertation dissertation_draft.tex
	cd $(OUT)/dissertation && BIBINPUTS=../../latex: $(BIBTEX) dissertation_draft
	cd dissertation && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/dissertation dissertation_draft.tex
	cd dissertation && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/dissertation dissertation_draft.tex

manual:
	@mkdir -p $(OUT)/manual
	cd manual && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/manual step_by_step_manual.tex
	cd $(OUT)/manual && BIBINPUTS=../../latex: $(BIBTEX) step_by_step_manual
	cd manual && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/manual step_by_step_manual.tex
	cd manual && TEXINPUTS=../latex: $(PDFLATEX) -interaction=nonstopmode -halt-on-error -output-directory=../$(OUT)/manual step_by_step_manual.tex

clean:
	rm -rf $(OUT)

