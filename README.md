# Redes_neurais

Este repositório demonstra implementações básicas de **Perceptrons** e **Adaline** em JavaScript e Python, e inclui exemplos de aplicação em diversos conjuntos de dados (flores, diabetes). O objetivo é ilustrar o funcionamento interno desses algoritmos de aprendizagem supervisionada passo a passo.

---

## 📖 Descrição dos Algoritmos

---

### Perceptron

- **Tipo**: Classificador linear binário.  
- **Como funciona**: ajusta pesos `w` e bias `b` para minimizar erros de classificação. A cada iteração, para cada amostra **x**, calcula:
  
  $$y_{\text{pred}} = \mathrm{sign}(w \cdot x + b)$$

  e atualiza:

  $$
  \begin{aligned}
    w &\leftarrow w + \eta\,(y_{\text{true}} - y_{\text{pred}})\,x,\\
    b &\leftarrow b + \eta\,(y_{\text{true}} - y_{\text{pred}})
  \end{aligned}
  $$

  onde `η` é a taxa de aprendizado.

- **Implementação** (`perceptron.js`):
  1. Classe `Perceptron` com métodos `train()` e `predict()`.  
  2. Exemplo de uso no final do arquivo.

---

### Adaline (Adaptive Linear Neuron)

- **Tipo**: Rede de camada única com saída contínua e função de ativação linear.
- **Como funciona**: minimiza o erro quadrático médio (MSE)
  
  $$
  J(w,b) = \frac{1}{2N} \sum_{i=1}^N \bigl(w \cdot x^{(i)} + b - y^{(i)}\bigr)^2
  $$

  usando **gradiente descendente**. As atualizações são:

  $$
  \begin{aligned}
    w &\leftarrow w - \eta \,\frac{\partial J}{\partial w},\\
    b &\leftarrow b - \eta \,\frac{\partial J}{\partial b}.
  \end{aligned}
  $$

- **Implementação** (`adalaine.py`):
  1. Classe `Adaline` com métodos:
     - `fit()`  
     - `net_input()`  
     - `activation()`  
     - `predict()`  
  2. Registro de histórico de custo por época para análise de convergência.

- **Script de exemplo** (`flower_adalaine.py`):
  1. Carrega `dataFlower.csv` (ex.: Iris simplificado).  
  2. Normaliza características.  
  3. Treina Adaline e plota a evolução do custo ao longo das épocas.

---
