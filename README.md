# 🦴 Classificação de Fraturas Ósseas com Redes Neurais Convolucionais (CNN)

## 📖 Descrição

Este repositório contém o código-fonte do projeto final da disciplina de **Inteligência Artificial**. O objetivo do trabalho é utilizar técnicas de classificação de imagens para resolver um problema real, treinando um modelo de aprendizado de máquina para a tarefa.

O problema escolhido foi a **detecção de fraturas em imagens de raios-x**, um desafio relevante na área médica onde a automação pode auxiliar na velocidade e precisão do diagnóstico. O projeto utiliza uma **Rede Neural Convolucional (CNN)** para classificar as imagens em duas categorias: **fraturado** e **não fraturado**.

---

## 📋 Tabela de Conteúdos

- [Descrição](#descrição)
- [Funcionalidades](#️funcionalidades)
- [Conjunto de Dados](#conjunto-de-dados)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Resultados](#resultados)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Instalação e Uso](#instalação-e-uso)
- [Autores](#autores)

---

## ✨ Funcionalidades

- **Treinamento de um Modelo CNN**: Utiliza um modelo de aprendizado profundo (deep learning) para aprender a diferenciar imagens com e sem fraturas.
- **Pré-processamento de Imagens**: Redimensionamento, normalização e verificação de integridade dos arquivos.
- **Avaliação de Desempenho**: Métricas como acurácia, precisão, recall e F1-score.
- **Visualização de Resultados**: Geração de matriz de confusão e exemplos de predições.

---

## 📊 Conjunto de Dados

O projeto utiliza o conjunto de dados público **Fracture-Multi Region X-Ray Data**, disponível na plataforma Kaggle. Este dataset é composto por imagens de raios-x divididas em duas classes, já separadas em conjuntos de treinamento, validação e teste.

🔗 [Fracture-Multi Region X-Ray Data (Kaggle)](https://www.kaggle.com/datasets/raddar/osteoporotic-fracture-dataset)

---

## 🚀 Tecnologias Utilizadas

- Python 3.11
- TensorFlow (Keras)
- Scikit-learn
- Numpy
- Matplotlib
- Seaborn

---

## 🏆 Resultados

O modelo treinado alcançou **99% de acurácia** no conjunto de teste.

### 📄 Relatório de Classificação:

```
               precision    recall  f1-score   support

    fractured       1.00      0.99      0.99       238
not fractured       0.99      1.00      0.99       262

     accuracy                           0.99       500
    macro avg       0.99      0.99      0.99       500
 weighted avg       0.99      0.99      0.99       500
```

### 📉 Matriz de Confusão

A matriz de confusão abaixo ilustra o baixo número de erros (apenas 3 em 500 imagens):

![Matriz de Confusão](https://github.com/ViniciusSantos-jpg/IA-projeto_fraturas_cnn/blob/main/Figure_1.png)

---

## 📁 Estrutura do Projeto

```
.
├── preprocess.py              # Funções de pré-processamento
├── model.py                   # Arquitetura da CNN
├── train.py                   # Treinamento do modelo
├── evaluate.py                   # Avaliação do modelo treinado
├── fracture_detection_model.keras  # Modelo salvo
├── requirements.txt              # Dependências do projeto
└── README.md                     # Documentação
```

---

## 🛠️ Instalação e Uso

### 1. Clone o Repositório

```bash
git clone https://github.com/ViniciusSantos-jpg/IA-projeto_fraturas_cnn
cd IA-projeto_fraturas_cnn
```

### 2. Crie um Ambiente Virtual (Recomendado)

```bash
python3 -m venv tf-env
source tf-env/bin/activate  # No Linux/macOS
# tf-env\Scripts\activate   # No Windows
```

### 3. Instale as Dependências

```bash
pip install -r requirements.txt
```

### 4. Baixe o Conjunto de Dados

- Faça o download do dataset no [Kaggle](https://www.kaggle.com/datasets/raddar/osteoporotic-fracture-dataset)
- Descompacte e organize conforme esperado pelos scripts (veja a variável `DATASET_BASE_DIR`).

### 5. Execute os Scripts

```bash
# Para treinar o modelo
python train.py

# Para avaliar o modelo treinado
python evaluate.py
```

---

## 👨‍💻 Autores

Este projeto foi desenvolvido por:

- Vinícius Santos Silva


