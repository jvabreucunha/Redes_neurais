# Soma dos erros quadraticos  
# J(w) = 1/2 * i(Σ) (Yi - Yî)²

data = [[0, 0, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 0],
        [1, 1, 0, 0]]

w = [0.1, 0.1, 0.1, 0.1]
learning_rate = 0.1

X_1 = 0
X_2 = 1
X_3 = 2
Y = 3

def Z(S):
    return w[0] + w[1] * data[S][0] + w[2] * data[S][1] + w[3] * data[S][2]

def phi(z):
    if z > 0:
        return 1
    else:
        return 0

def J():
    return 1/2 * ((data[0][-1] - phi(Z(0)))**2 + (data[1][-1] - phi(Z(1)))**2 + (data[2][-1] - phi(Z(2)))**2 + (data[3][-1] - phi(Z(3)))**2)

print("---1º Época---")
print("Erro J(w) =", J(), '\n')

def delta_j(j):
  if j == 0:
    return learning_rate * (
        ((data[0][-1] - phi(Z(0))) * 1) + \
        ((data[1][-1] - phi(Z(1))) * 1) + \
        ((data[2][-1] - phi(Z(2))) * 1) + \
        ((data[3][-1] - phi(Z(3))) * 1))
  elif j == 1:
    return learning_rate*(
        ((data[0][-1] - phi(Z(0))) * data[0][X_1]) + \
        ((data[1][-1] - phi(Z(1))) * data[1][X_1]) + \
        ((data[2][-1] - phi(Z(2))) * data[2][X_1]) + \
        ((data[3][-1] - phi(Z(3))) * data[3][X_1]))
  elif j == 2:
    return learning_rate*(
        ((data[0][-1] - phi(Z(0))) * data[0][X_2]) + \
        ((data[1][-1] - phi(Z(1))) * data[1][X_2]) + \
        ((data[2][-1] - phi(Z(2))) * data[2][X_2]) + \
        ((data[3][-1] - phi(Z(3))) * data[3][X_2]))
  elif j == 3:
    return learning_rate*(
        ((data[0][-1] - phi(Z(0))) * data[0][X_3]) + \
        ((data[1][-1] - phi(Z(1))) * data[1][X_3]) + \
        ((data[2][-1] - phi(Z(2))) * data[2][X_3]) + \
        ((data[3][-1] - phi(Z(3))) * data[3][X_3]))  
    
is_balanced = False
interaction = 2
while not is_balanced:
    print("---",interaction,"º Época---")
    print("Erro J(w) =", J())

    aux_0 = w[0] + delta_j(0)
    aux_1 = w[1] + delta_j(1)
    aux_2 = w[2] + delta_j(2)
    aux_3 = w[3] + delta_j(3)

    w = [aux_0, aux_1, aux_2, aux_3]
    print("w =", w)
    print("\n")
    
    interaction += 1
    
    value_J = J()
    if value_J == 0:
        print('finalizado')
        print("Erro J(w) =", J()) 
        print("w =", w)
        break