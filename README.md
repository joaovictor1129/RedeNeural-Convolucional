# Detecção de Pneumonia em Imagens de Raios-X do Tórax

## Visão Geral
Este projeto utiliza técnicas de aprendizado profundo, especificamente Redes Neurais Convolucionais (CNN), para detectar automaticamente a pneumonia a partir de imagens de raios-X do tórax. A solução aproveita o TensorFlow e o Keras para a construção, treinamento e avaliação do modelo.

## Conjunto de Dados
O conjunto de dados consiste em imagens de raios-X do tórax categorizadas em conjuntos de treinamento, validação e teste. As imagens são organizadas em diretórios respectivos (`train`, `test`, `val`).

## Funcionalidades
- **Análise Automatizada de Imagens**: Classifica imagens de raios-X em normais ou indicativas de pneumonia.
- **Aumento de Dados**: Aprimora o conjunto de dados por meio de transformações como zoom, cisalhamento e inversão horizontal.
- **Modelo CNN**: Um modelo CNN personalizado para classificação binária.
- **Aprendizado por Transferência**: Utiliza o modelo pré-treinado InceptionV3 para maior precisão com menos amostras de treinamento.
- **Métricas de Avaliação**: Métricas de acurácia e perda para avaliação de desempenho.

## Instalação
Certifique-se de ter os seguintes pré-requisitos instalados:
- Python 3.x
- TensorFlow 2.x
- Numpy
- Scipy
- Matplotlib
- Jupyter Notebook (opcional, para execução interativa)

Instale as bibliotecas necessárias usando:
```bash
pip install tensorflow numpy scipy matplotlib
