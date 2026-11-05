# ADR 005 — Protocolo de treinamento

## Decisão

Congelar `configs/lee_synthetic_final.yaml` como protocolo de produção:

- 75.000/10.000/15.000 trajetórias para treino/validação/teste;
- sequências exclusivamente com 1.000 eventos;
- 300 épocas máximas, Adam, MSE e early stopping com paciência 30;
- cinco seeds de treino;
- três estratos balanceados do branching ratio;
- MLE heurístico com cinco starts, sem inicialização-oráculo.

Uma configuração separada, `configs/local_smoke.yaml`, valida a integração sem gerar
evidência final. Seus números não devem ser comparados como resultado científico.

## Artefatos obrigatórios

Cada execução salva manifestos/checksums dos shards, configuração resolvida,
ambiente, commit, histórico por época, checkpoint de melhor validação, métricas por
parâmetro, erro de `g`, validade, convergência MLE e latência mediana/p95.

## Política de teste

Dados sintéticos de teste não participam da escolha de época. No estudo empírico,
somente `calibration_train` ajusta pesos; `validation` escolhe a época; `test_closed`
exige `--open-test`. Outubro de 2023 é robustez temporal e não permite retreino.
