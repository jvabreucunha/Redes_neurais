import pandas as pd

data = pd.read_csv("diabetes.csv").values

w = [-14545.099999999837, 22485.099999999824, 2617.19999999975, -6264.700000000705, -332.80000000004566, 112.99999999970896, 1048.079999999461, 3398.6795000000006, -1991.100000000028]
learning_rate = 0.10

def Z(S):
    z = w[0]  # Bias
    for j in range(1, len(w)):
        z += w[j] * data[S][j-1]
    return z

# Função de ativação
def phi(z):
    return 1 if z > 0 else 0

def J():
    total_error = 0
    for i in range(len(data)):
        total_error += (data[i][-1] - phi(Z(i))) ** 2
    return total_error / 2

# Função para calcular a variação do peso w[j] considerando todas as amostras
def delta_j(j):
    total_delta = 0
    for i in range(len(data)):
        error = data[i][-1] - phi(Z(i))
        if j == 0:
            total_delta += error  # Bias
        else:
            total_delta += error * data[i][j-1]
    return learning_rate * total_delta

# Função para calcular a precisão
def calculate_precision():
    hits = 0
    total = len(data)
    for i in range(total):
        predicao = phi(Z(i))
        if predicao == data[i][-1]:
            hits += 1
    precision = hits / total * 100
    return precision

maiorAcerto = 0
peso_usado = []

# Treinamento da rede
interaction = 1
for epoch in range(1000):
    print(f"--- {interaction}º Época ---")
    print("Erro J(w) =", J())
    
    new_w = [w[j] + delta_j(j) for j in range(len(w))]
    w = new_w  # Atualiza os pesos

    print("w =", w)
    hit_rate = calculate_precision()
    print(f"Taxa de acerto da rede: {hit_rate}%\n")

    interaction += 1
    if hit_rate > maiorAcerto:
        maiorAcerto = hit_rate
        peso_usado = new_w

print(f"Maior taxa de Acerto: {maiorAcerto}")
print(peso_usado)

#Maxio de acerto algo em torno de 71%