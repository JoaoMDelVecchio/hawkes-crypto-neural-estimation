# Dados

O conteúdo desta pasta não é versionado. Apenas este guia e os arquivos `.gitkeep`
permanecem no Git.

- `raw/`: arquivos originais e `manifest.jsonl`;
- `interim/`: arquivos extraídos e normalizados;
- `processed/`: sequências de eventos e splits;
- `external/`: fontes auxiliares que não sejam Binance.

Todo arquivo bruto deve ser recuperável por URL e SHA-256. Não edite dados em `raw/`.
Transformações devem escrever em uma camada posterior e registrar a configuração usada.
