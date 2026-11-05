# Perguntas e hipóteses de pesquisa

## Pergunta principal

Mecanismos de atenção melhoram a estimação amortizada dos parâmetros de um Hawkes 2D
em movimentos de preço de cripto quando comparados à LSTM de Lee (2023)?

## Perguntas secundárias

1. Dados públicos da Binance permitem reconstruir, com qualidade suficiente, a entrada
   de alta frequência usada no artigo?
2. Qual mercado — spot ou futuro perpétuo — produz sequências mais adequadas?
3. A implementação da LSTM reproduz a relação entre acurácia e custo reportada?
4. Resultados sintéticos se mantêm quando a rede imita MLE em dados reais?
5. O ganho ou perda do Transformer depende do comprimento ou do regime de clustering?

## Hipóteses

- **H1:** ao menos um dos mercados oferece segmentos contínuos suficientes após a
  filtragem em 0,1 s.
- **H2:** a LSTM tem erro da mesma ordem do MLE e inferência substancialmente menor.
- **H3:** o Transformer reduz o MSE de teste ou aumenta a robustez em sequências com
  dependências longas, sob protocolo idêntico.

H3 pode ser rejeitada; um resultado negativo bem controlado permanece uma contribuição.

## Fora do escopo

Livro de ofertas completo, previsão direcional, estratégia de trading, execução de ordens
e alegações causais sobre formação de preço.
