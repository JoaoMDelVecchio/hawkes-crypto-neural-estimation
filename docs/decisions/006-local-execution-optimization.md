# ADR 006 — Otimização da execução local

## Contexto

A primeira execução serial consumiu mais de sete horas sem completar a primeira seed.
O código sincronizava o MPS em cada batch ao converter a soma de erros para `float` e
enviava shards inteiros de validação/teste ao dispositivo.

## Decisão

Usar `configs/lee_local_optimized.yaml`, que mantém dataset, splits, seeds, Adam,
learning rate, máximo de 300 épocas, paciência 30 e batch efetivo. A execução muda
somente a mecânica:

- dados de cada split são carregados uma vez em RAM;
- treino, validação e teste usam minibatches;
- a soma de erros permanece no dispositivo até o fim da época;
- checkpoints atômicos preservam modelo, Adam, época, early stopping e RNG;
- cada combinação modelo/seed é um job retomável;
- `torch.compile` é usado onde o benchmark mostrou ganho;
- o Transformer de 1.000 eventos usa batch físico 4 com gradient accumulation para
  preservar batch efetivo 8;
- o MLE usa um pool persistente de quatro processos, reutilizado entre chunks, e uma
  thread BLAS por worker;
- a log-verossimilhança do MLE calcula valor e gradiente analítico na mesma avaliação;
  as recorrências exponenciais são resolvidas como sistemas bidiagonais, sem diferenças
  finitas sobre os quatro parâmetros.

## Benchmark no Mac

O MPS venceu CPU na LSTM de 1.000 eventos, e `torch.compile` elevou o throughput do
Transformer nesse comprimento. Executar LSTM na CPU simultaneamente reduziu o
throughput do Transformer em mais de 50%; portanto, o perfil de menor tempo total usa
um único job MPS por vez.

No benchmark do MLE com 16 trajetórias, quatro workers concluíram em 16,04 s e oito
workers em 16,16 s. O perfil escolhe quatro: foi marginalmente mais rápido e reduz a
contenção de CPU e memória. Manter o pool vivo entre chunks evita repetir o custo de
criação dos processos. Esse benchmark precede o gradiente analítico; a implementação
final preserva a mesma log-verossimilhança, verificada por testes contra o cálculo
direto e por diferenças finitas do gradiente.

A execução integral com gradiente analítico ajustou as 15.000 trajetórias em
1.574,61 s de parede, ou 9,53 trajetórias/s com quatro workers, e obteve 100% de
convergência numérica.

Mixed precision FP16 não melhorou o throughput e foi rejeitada para evitar mudança
numérica sem benefício. A diferença observada entre inferência compilada e eager no
smoke test foi inferior a \(2\times10^{-8}\) em termos relativos.

## Contexto histórico supersedido

O perfil anterior também avaliava trajetórias de 2.000 eventos. Nesse comprimento,
`torch.compile` aproximadamente dobrou o throughput do Transformer, e batch físico 2
atingiu cerca de 42 amostras/s, contra 18 amostras/s no batch 8. Um smoke test
histórico com split 48/16/16 e duas épocas registrou MSE 0,3193 para LSTM, 0,0960 para
Transformer e 0,0174 para MLE, com 100% de validade neural; o MLE foi medido em apenas
duas trajetórias. O perfil anterior também especificava oito workers de MLE.

Esses números documentam somente a otimização e a integração antigas. O comprimento
de 2.000 eventos e a configuração de oito workers estão supersedidos e não fazem
parte do protocolo final nem constituem instrução de execução.

## Consequências

O paralelismo máximo seguro não corresponde ao maior número de processos: CPU e GPU
compartilham memória e largura de banda. O scheduler serializa jobs MPS e paraleliza
somente o MLE, cuja unidade de trabalho é independente. A execução interrompida foi
arquivada e não participa das tabelas finais.

> **Nota (30/10/2026).** O tempo de 1.574,61 s para as 15.000 trajetórias foi medido
> com o MLE concorrendo com o treino na GPU. Em benchmark controlado, o mesmo código
> (que reproduz exatamente `records.jsonl`) levou 2,5 ms de parede por trajetória com
> quatro processos e 9,6 ms em um núcleo. Ver [ADR 007](007-analise-complementar.md).
