# Projeto de Redes Neurais e Machine Learning

Este projeto é uma imersão em algoritmos clássicos de **machine learning** e redes neurais, com implementações detalhadas do **Perceptron**, **Adaline**, e uma exploração do funcionamento de redes neurais modernas. Abaixo, descrevemos os conceitos técnicos, matemática subjacente e aplicações práticas.

---

## 🧠 Conceitos Fundamentais

### 1. Machine Learning (ML)
- **Definição**: Capacidade de sistemas aprenderem padrões a partir de dados sem programação explícita.
- **Tipos**:
  - **Supervisionado**: Treinamento com pares (entrada, saída). Ex: Classificação, regressão.
  - **Não supervisionado**: Identificação de estruturas em dados não rotulados. Ex: Clustering.
  - **Reforço**: Aprendizado por interação e recompensa.

---

## 🔍 Perceptron: O Bloco Básico

### Visão Geral
- Desenvolvido por Frank Rosenblatt (1957), é a base para redes neurais.
- **Objetivo**: Classificação binária (ex: separar duas classes linearmente).

### Funcionamento Matemático
1. **Entrada**: Vetor de características \( \mathbf{x} = [x_1, x_2, ..., x_n] \).
2. **Pesos**: \( \mathbf{w} = [w_1, w_2, ..., w_n] \) (inicializados aleatoriamente).
3. **Saída Bruta**: \( z = \mathbf{w} \cdot \mathbf{x} + b \) (bias: \( b \)).
4. **Função de Ativação**: Degrau (Step Function):
   \[
   y_{\text{pred}} = 
   \begin{cases} 
   1 & \text{se } z \geq 0, \\
   0 & \text{se } z < 0.
   \end{cases}
   \]

### Atualização dos Pesos
- **Regra de Aprendizado**: 
  \[
  \Delta w_i = \alpha \times (y_{\text{true}} - y_{\text{pred}}) \times x_i
  \]
  - \( \alpha \): Taxa de aprendizado (controla a magnitude da atualização).
  - Iteração até convergência ou máximo de épocas.

### Limitações
- Só resolve problemas **linearmente separáveis**.
- Não lida com ruídos ou dados não lineares.

---

## 📈 Adaline (Adaptive Linear Neuron)

### Evolução do Perceptron
- Introduzido por Bernard Widrow e Ted Hoff (1960).
- **Diferença Chave**: Minimiza uma função de custo contínua (erro quadrático) em vez de atualizações diretas.

### Funcionamento
1. **Saída Bruta**: \( z = \mathbf{w} \cdot \mathbf{x} + b \).
2. **Função de Ativação**: Linear (identidade) para treinamento.
3. **Função de Custo**: Erro Quadrático Médio (MSE):
   \[
   J(\mathbf{w}) = \frac{1}{2} \sum_{i=1}^{m} (y_{\text{true}}^{(i)} - z^{(i)})^2
   \]

### Gradiente Descendente
- **Atualização de Pesos**:
  \[
  \mathbf{w} := \mathbf{w} - \alpha \nabla J(\mathbf{w})
  \]
  - Gradiente calculado como:
    \[
    \nabla J(\mathbf{w}) = -\sum_{i=1}^{m} (y_{\text{true}}^{(i)} - z^{(i)}) \mathbf{x}^{(i)}
    \]

### Vantagens sobre o Perceptron
- Suaviza a função de custo, permitindo convergência mais estável.
- Pode ser estendido para problemas não lineares com kernels.

---

## 🚀 Redes Neurais Modernas: Arquitetura e Treinamento

### Estrutura Hierárquica
1. **Camada de Entrada**: 
   - Neurônios correspondem às features dos dados (ex: pixels em uma imagem).
2. **Camadas Ocultas**:
   - Extraem características intermediárias (ex: bordas em visão computacional).
3. **Camada de Saída**:
   - Neurônios correspondem às classes ou valores preditos (ex: softmax para classificação).

### Neurônio Artificial Moderno
- **Função de Ativação**: 
  - **ReLU**: \( f(z) = \max(0, z) \) (evita vanishing gradient).
  - **Sigmoid**: \( f(z) = \frac{1}{1 + e^{-z}} \) (para probabilidades).
  - **Softmax**: Normaliza saídas para distribuição de probabilidade.

### Algoritmo de Treinamento
1. **Forward Propagation**:
   - Cálculo da saída da rede: \( \hat{y} = f(\mathbf{W}^{(L)} \cdot f(\mathbf{W}^{(L-1)} \cdot ... f(\mathbf{W}^{(1)} \mathbf{x}))) \).
2. **Função de Custo**:
   - **Cross-Entropy**: Para classificação.
     \[
     J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{c}^{(i)} \log(\hat{y}_{c}^{(i)})
     \]
   - **MSE**: Para regressão.
3. **Backpropagation**:
   - Cálculo de gradientes usando a regra da cadeia:
     \[
     \frac{\partial J}{\partial w_{ij}^{(k)}} = \frac{\partial J}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z^{(k)}} \cdot \frac{\partial z^{(k)}}{\partial w_{ij}^{(k)}}
     \]
4. **Otimização**:
   - **Adam**: Combina momentum e adaptação de taxa de aprendizado.
   - **SGD com Momentum**: Acelera convergência em direções consistentes.

### Técnicas Avançadas
- **Regularização**:
  - **L1/L2**: Penaliza pesos grandes.
  - **Dropout**: Desativa neurônios aleatoriamente durante o treino.
- **Batch Normalization**: Normaliza saídas das camadas para acelerar treinamento.

---

## 📊 Aplicações do Projeto

### Implementações
1. **Perceptron**:
   - Classificação de dados sintéticos linearmente separáveis.
   - Visualização da fronteira de decisão.
2. **Adaline**:
   - Regressão linear com gradiente descendente.
   - Análise de sensibilidade à taxa de aprendizado.
3. **Redes Neurais**:
   - Classificação de MNIST ou Iris dataset com PyTorch/TensorFlow.
   - Experimentos com diferentes funções de ativação e otimizadores.

### Análise de Resultados
- **Métricas**:
  - Acurácia, Precisão, Recall, F1-Score.
  - Matriz de Confusão e Curva ROC.
- **Visualização**:
  - Gráficos de convergência (loss vs. época).
  - Mapas de características em camadas ocultas.

---

## 🔧 Funcionamento Básico de uma Rede Neural Moderna (Exemplo)

```python
# Exemplo Simplificado em PyTorch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(784, 128)  # Input -> Hidden
        self.layer2 = nn.Linear(128, 10)    # Hidden -> Output
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.layer2(x)
        return x

# Treinamento
model = NeuralNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for data, labels in dataloader:
        outputs = model(data)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
