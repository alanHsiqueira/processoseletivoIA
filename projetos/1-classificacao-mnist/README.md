# Relatório Técnico - Projeto 1: MNIST

## 1. Bibliotecas e Versões
- **Python:** 3.11.15
- **TensorFlow:** 2.21.0
- **NumPy:** 2.4.6

## 2. Arquitetura do Modelo
O modelo foi implementado como uma CNN sequencial, adequada para extrair padrões espaciais em imagens. A estrutura adotada foi:
- **3 blocos convolucionais:** cada bloco usa `Conv2D` com 32, 64 e 128 filtros, seguido de `BatchNormalization` e `MaxPooling2D`.
- **Camada final:** após `Flatten`, foi incluído `Dropout` antes da saída `Dense` com 10 neurônios e ativação `softmax`.

## 3. Hiperparâmetros e Justificativas
- **Dropout (0.5):** aplicado antes da camada de saída para reduzir a chance de *overfitting* e forçar a rede a aprender representações mais estáveis.
- **Early Stopping:** configurado com `patience=3` e monitoramento de `val_loss` para interromper o treino assim que o desempenho na validação deixasse de melhorar.

## 4. Métrica de Avaliação
- **Acurácia Final no Conjunto de Validação:** 98.62%

## 5. Otimização para Edge AI (TFLite)
- **Técnica Utilizada:** aplicação da otimização padrão do conversor por meio de `tf.lite.Optimize.DEFAULT` antes da conversão.
- **Tamanho do modelo original (`model.h5`):** 1.2 MB
- **Tamanho do modelo otimizado (`model.tflite`):** 104 KB
- **Discussão:** A redução foi de mais de 10 vezes no tamanho em disco, saindo de cerca de 1200 KB para 104 KB. Isso torna o artefato mais adequado para cenários com memória RAM/Flash limitada.

## 6. Inferência no Edge (Teste Prático)
Saída gerada pelo terminal durante a inferência:
```text
Rodando inferencia em 5 amostras usando model.tflite:

Amostra 1: predito=7 | real=7
Amostra 2: predito=2 | real=2
Amostra 3: predito=1 | real=1
Amostra 4: predito=0 | real=0
Amostra 5: predito=4 | real=4
```
**Comentário sobre o resultado:** O modelo em formato `.tflite` manteve o comportamento esperado na inferência. Nas 5 amostras de teste exibidas acima, todas as previsões coincidiram com os rótulos reais, sem sinal de perda perceptível de desempenho após a otimização.
