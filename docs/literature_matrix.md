# Matriz de literatura

Chaves BibTeX em `thesis/latex/references.bib`. PDFs locais em `references/` quando
indicados.

## Base do trabalho

| Referência | Chave | Papel no trabalho | Local |
|---|---|---|---|
| Lee (2023) | `lee2023` | paper-base: LSTM para estimação amortizada; tabela do ponto fixo (E2) | `references/neural_estimation/lee_2023_rnn_parameter_estimation.pdf` |
| Lee e Seo (2017) | `leeseo2017` | Hawkes marcado de preço e volatilidade | — |
| Lee (2024) | `lee2024` | volatilidade de Hawkes em preço filtrado | `references/hawkes/lee_2024_hawkes_volatility.pdf` |
| Hawkes (1971a, 1971b) | `hawkes1971`, `hawkes1971b` | processos auto e mutuamente excitantes | `references/hawkes/hawkes_1971_biometrika.pdf` |

## Hawkes em finanças

| Referência | Chave | Papel |
|---|---|---|
| Bacry et al. (2013) | `bacry2013` | Hawkes 2D de altas e baixas; excitação cruzada como reversão |
| Bacry, Mastromatteo e Muzy (2015); Hawkes (2018) | `bacry2015`, `hawkes2018` | revisões do campo |
| Bowsher (2007); Embrechts, Liniger e Lin (2011) | `bowsher2007`, `embrechts2011` | modelos multivariados de eventos de mercado |
| Filimonov e Sornette (2012); Hardiman, Bercot e Bouchaud (2013) | `filimonov2012`, `hardiman2013` | razão de ramificação e reflexividade |
| Aït-Sahalia, Cacho-Diaz e Laeven (2015) | `aitsahalia2015` | contágio com processos mutuamente excitantes |
| Muni Toke (2011) | `mutoke2011` | introdução didática (`references/hawkes/hawkes_course_slides.pdf`) |
| Primo (2018); Silva (2026) | `primo2018`, `silva2026` | dissertações brasileiras com MLE e diagnósticos |

## Estimação, simulação e diagnóstico

| Referência | Chave | Papel |
|---|---|---|
| Ozaki (1979); Ogata (1978) | `ozaki1979`, `ogata1978` | MLE com kernel exponencial e propriedades assintóticas |
| Ogata (1981) | `ogata1981` | simulação por *thinning* |
| Ogata (1988); Brown et al. (2002) | `ogata1988`, `brown2002` | resíduos reescalados e *time-rescaling* |
| Daley e Vere-Jones (2003); Laub, Lee e Taimre (2021) | `daley2003`, `laub2021` | teoria de processos pontuais; livro sobre Hawkes |

## Inferência amortizada e estimadores neurais

| Referência | Chave | Papel |
|---|---|---|
| Cranmer, Brehmer e Louppe (2020) | `cranmer2020` | inferência baseada em simulação |
| Sainsbury-Dale, Zammit-Mangion e Huser (2024); Lenzi et al. (2023); Gerber e Nychka (2021) | `sainsburydale2024`, `lenzi2023`, `gerber2021` | estimadores neurais como estimadores de Bayes |
| Radev et al. (2022) | `radev2022` | posterior amortizada (BayesFlow) |
| Lambe et al. (2025) | `lambe2025` | Hawkes observado por contagens; verossimilhança intratável (`references/neural_estimation/`) |
| Lehmann e Casella (1998) | `lehmann1998` | média a posteriori minimiza o risco de Bayes quadrático |

## Processos pontuais neurais e arquiteturas

| Referência | Chave | Papel |
|---|---|---|
| Du et al. (2016); Mei e Eisner (2017) | `du2016`, `mei2017` | intensidade modelada por RNN |
| Zuo et al. (2020); Zhang et al. (2020) | `zuo2020`, `zhang2020` | Hawkes com atenção |
| Shchur et al. (2021) | `shchur2021` | revisão de processos pontuais neurais |
| Hochreiter e Schmidhuber (1997); Cho et al. (2014) | `hochreiter1997`, `cho2014` | LSTM e GRU |
| Vaswani et al. (2017); Xiong et al. (2020) | `vaswani2017`, `xiong2020` | Transformer; posição da normalização e aquecimento |
| Kingma e Ba (2015) | `kingma2015` | Adam |

## Avaliação estatística

| Referência | Chave | Papel |
|---|---|---|
| Bouthillier et al. (2021) | `bouthillier2021` | variância entre seeds em comparações de ML |
| Diebold e Mariano (1995); Efron e Tibshirani (1993) | `diebold1995`, `efron1993` | comparação pareada de erros; *bootstrap* |

## Alta frequência, volatilidade e cripto

| Referência | Chave | Papel |
|---|---|---|
| Bouchaud et al. (2018) | `bouchaud2018` | microestrutura (`references/high_frequency/`) |
| Bauwens e Giot (2001); Pacurar (2008); Morettin (2007) | `bauwens2001`, `pacurar2008`, `morettin2007` | atividade intradiária e modelos de duração |
| Andersen et al. (2003); Barndorff-Nielsen e Shephard (2002); Zhang, Mykland e Aït-Sahalia (2005) | `andersen2003`, `barndorff2002`, `zhang2005` | *realized volatility* e ruído de microestrutura |
| Taddeo, Salvi e Rivera-Castro (2024) | `taddeo2024` | Hawkes para saltos em cripto (Binance, 1 min) |
| Makarov e Schoar (2020); He et al. (2022) | `makarov2020`, `he2022perpetual` | estrutura do mercado de cripto; futuros perpétuos |
| Binance Public Data | `binancePublicData` | fonte dos dados |
