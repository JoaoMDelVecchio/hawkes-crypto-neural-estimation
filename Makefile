.PHONY: test smoke synthetic-smoke synthetic-final train-final benchmark-local train-local-optimized mle-local consolidate-local figures-local local-final calibration empirical-labels-open empirical-test robustness calibration-full fixed-point posthoc diagnostics empirical-labels empirical-models empirical-fragment volatility robustness-scope-a thesis-en-fragments compare-finetune addendum-beta addendum-train addendum-select addendum-holdout addendum-grid addendum-fragment scope-b grid-correction volatility-addendum thesis-preliminar

test:
	uv run ruff check src/hawkes_crypto scripts tests
	uv run pytest

synthetic-smoke:
	uv run python scripts/generate_synthetic_dataset.py --config configs/local_smoke.yaml

smoke: synthetic-smoke
	uv run python scripts/run_synthetic_experiment.py --config configs/local_smoke.yaml

synthetic-final:
	uv run python scripts/generate_synthetic_dataset.py --config configs/lee_synthetic_final.yaml

train-final:
	uv run python scripts/run_synthetic_experiment.py --config configs/lee_synthetic_final.yaml

benchmark-local:
	uv run python scripts/benchmark_local_training.py \
		--config configs/lee_local_optimized.yaml

train-local-optimized:
	uv run python scripts/run_local_scheduler.py \
		--config configs/lee_local_optimized.yaml

mle-local:
	uv run python scripts/run_mle_incremental.py \
		--config configs/lee_local_optimized.yaml --length 1000

consolidate-local:
	uv run python scripts/consolidate_local_results.py \
		--config configs/lee_local_optimized.yaml

figures-local:
	uv run python scripts/generate_final_figures.py

local-final:
	$(MAKE) train-local-optimized
	$(MAKE) mle-local
	$(MAKE) consolidate-local
	$(MAKE) figures-local
	uv run python scripts/generate_final_report_fragment.py
	$(MAKE) -C thesis -j3 all

calibration:
	uv run python scripts/run_calibration_mle.py --config configs/lee_synthetic_final.yaml

# Escopo empírico A. Etapa 4: rótulos MLE em treino e validação, redes treinadas em janelas
# reais, variante escolhida na validação; o teste continua fechado.
empirical-labels:
	uv run python scripts/run_calibration_mle.py --config configs/lee_local_optimized.yaml \
		--splits calibration_train validation \
		--output artifacts/empirical_mle/calibration_validation.json

empirical-models: empirical-labels
	uv run python scripts/run_empirical_models.py --config configs/lee_local_optimized.yaml \
		--mle-summary artifacts/empirical_mle/calibration_validation.json

# C4: volatilidade de Hawkes contra realized volatility em treino e validação (MLE e, se
# VARIANT for dado, o ensemble das redes do escopo A). O teste só na etapa 8.
volatility:
	uv run python scripts/run_hawkes_volatility.py --splits calibration_train validation \
		$(if $(VARIANT),--variant $(VARIANT))

# Fragmento LaTeX (inglês) com o diagnóstico e o escopo A; inclui o teste se já foi aberto.
empirical-fragment:
	uv run python scripts/generate_empirical_fragment.py

# Etapa 8, uma única vez: rotula o teste e avalia os pesos congelados, sem retreinar.
empirical-labels-open:
	uv run python scripts/run_calibration_mle.py --config configs/lee_local_optimized.yaml \
		--splits calibration_train validation test_closed --open-test \
		--output artifacts/empirical_mle/calibration_validation_test.json

empirical-test: empirical-labels-open
	uv run python scripts/run_empirical_models.py --config configs/lee_local_optimized.yaml \
		--mle-summary artifacts/empirical_mle/calibration_validation_test.json --open-test

# Pós-teste (post hoc): LSTM × Transformer com ajuste fino, pareado por janela.
compare-finetune:
	PYTHONPATH=src .venv/bin/python scripts/compare_scope_a_finetune.py

# Etapa 8: escopo A na janela de robustez de outubro/2023, com os pesos congelados.
robustness-scope-a:
	uv run python scripts/run_robustness_scope_a.py --config configs/lee_local_optimized.yaml

robustness:
	test -n "$(MODEL)" && test -n "$(CHECKPOINT)"
	uv run python scripts/run_robustness_eval.py --config configs/lee_synthetic_final.yaml \
		--model "$(MODEL)" --checkpoint "$(CHECKPOINT)" --length "$(or $(LENGTH),1000)"

# Análises complementares (30/10/2026): não retreinam nem abrem o teste empírico.
calibration-full:
	uv run python scripts/run_calibration_mle.py --config configs/lee_local_optimized.yaml \
		--splits calibration_train --output artifacts/empirical_mle/calibration_jan15-22.json

# Etapa 3: diagnóstico de ajuste do Hawkes nas janelas reais de calibração (~40 s).
diagnostics:
	uv run python scripts/run_real_data_diagnostics.py --config configs/lee_local_optimized.yaml \
		--calibration artifacts/empirical_mle/calibration_jan15-22.json

fixed-point:
	uv run python scripts/run_fixed_point_e2.py --config configs/lee_local_optimized.yaml

posthoc: calibration-full
	uv run python scripts/analyze_posthoc.py --config configs/lee_local_optimized.yaml \
		--out artifacts/final_hawkes/lee_local_optimized/posthoc --figures thesis/build/figures \
		--calibration artifacts/empirical_mle/calibration_jan15-22.json
	uv run python scripts/generate_posthoc_fragment.py \
		--report artifacts/final_hawkes/lee_local_optimized/posthoc/posthoc_sintetico.json

# Versões em inglês para a tese em thesis/tg: fragmentos em thesis/latex/en/ e figuras
# *_en.pdf em thesis/build/figures, renderizados só a partir dos artefatos salvos (não
# treina, não roda MLE nem E2). As versões em português não são tocadas.
thesis-en-fragments:
	uv run python scripts/generate_final_report_fragment.py --lang en
	uv run python scripts/generate_posthoc_fragment.py --lang en \
		--report artifacts/final_hawkes/lee_local_optimized/posthoc/posthoc_sintetico.json
	uv run python scripts/run_fixed_point_e2.py --render-only --lang en
	uv run python scripts/generate_final_figures.py --lang en
	uv run python scripts/analyze_posthoc.py --config configs/lee_local_optimized.yaml \
		--out artifacts/final_hawkes/lee_local_optimized/posthoc --figures thesis/build/figures \
		--calibration artifacts/empirical_mle/calibration_jan15-22.json --figures-only --lang en

# Adendo (ADR 009): teste de β, receita de ajuste fino e semana nova (5–11/fev/2024).
addendum-beta:
	PYTHONPATH=src .venv/bin/python scripts/run_beta_sweep.py
	PYTHONPATH=src .venv/bin/python scripts/run_beta_sweep.py --betas 6 8 \
		--output artifacts/addendum_009/beta_sweep_extension.json

# Fila de tarefas: rode quantos workers quiser ao mesmo tempo (um na MPS, os demais na CPU).
addendum-train:
	PYTHONPATH=src .venv/bin/python scripts/run_addendum_finetune.py --device $(or $(DEVICE),auto)

addendum-select:
	PYTHONPATH=src .venv/bin/python scripts/run_addendum_finetune.py --select

addendum-holdout:
	PYTHONPATH=src .venv/bin/python scripts/run_addendum_finetune.py --open-holdout

# Viés da grade de 100 ms no β do MLE (post hoc; não lê o test_closed).
addendum-grid:
	PYTHONPATH=src .venv/bin/python scripts/run_grid_bias.py

addendum-fragment:
	PYTHONPATH=src .venv/bin/python scripts/generate_addendum_fragment.py

# Escopo B reduzido (ADR 010): gera, treina (5 workers na CPU), seleciona e lê mar/2024 uma vez.
scope-b:
	PYTHONPATH=src .venv/bin/python scripts/run_scope_b.py --generate
	for i in 1 2 3 4 5; do PYTHONPATH=src .venv/bin/python scripts/run_scope_b.py & done; wait
	PYTHONPATH=src .venv/bin/python scripts/run_scope_b.py --select
	PYTHONPATH=src .venv/bin/python scripts/run_scope_b.py --open-holdout

# ADR 011: a rede aprende a desfazer o viés da grade? (usa os dados sintéticos do escopo B)
grid-correction:
	PYTHONPATH=src .venv/bin/python scripts/run_grid_correction.py --generate-test
	for i in 1 2 3 4 5; do PYTHONPATH=src .venv/bin/python scripts/run_grid_correction.py & done; wait
	PYTHONPATH=src .venv/bin/python scripts/run_grid_correction.py --evaluate

# ADR 011: volatilidade de Hawkes com as redes do adendo (fevereiro/2024 e validação).
volatility-addendum:
	PYTHONPATH=src .venv/bin/python scripts/run_volatility_addendum.py

# Versão preliminar da monografia (para a banca): sem o verso da folha de rosto.
PDFLATEX ?= $(shell command -v pdflatex || echo $(HOME)/Library/TinyTeX/bin/universal-darwin/pdflatex)
# Usa a bibliografia já compilada da versão final (thesis/build/tg/main.bbl).
thesis-preliminar:
	mkdir -p thesis/build/tg_preliminar/chapters
	cp thesis/build/tg/main.bbl thesis/build/tg_preliminar/main_preliminar.bbl
	cd thesis/tg && for i in 1 2; do TEXINPUTS=.:../latex//: $(PDFLATEX) -interaction=nonstopmode \
		-halt-on-error -output-directory=../build/tg_preliminar -jobname=main_preliminar \
		'\def\itapreliminar{}\input{main}' > /dev/null; done
