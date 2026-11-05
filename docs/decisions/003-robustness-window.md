# ADR 003 — Janela de robustez

Status: **executada**  
Data: 2026-08-31

Foi processada a janela separada de 2023-10-23 a 2023-10-29 UTC, sem influência na
seleção de arquitetura ou nos hiperparâmetros do período primário.

Todos os 28 arquivos esperados foram obtidos (2 símbolos × 2 produtos × 7 dias).
O processamento de `bookTicker` gerou 14 partições:

- BTCUSDT: 652.773 eventos; mínimo diário de 22.480;
- ETHUSDT: 635.579 eventos; mínimo diário de 28.785;
- total: 1.288.352 eventos;
- 1.281 segmentos de 1.000 eventos.

Os artefatos ficam em `data/processed_robustness/` e não entram no Git.
