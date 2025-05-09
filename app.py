# Criando a base de dados
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
ages = np.random.randint( low=15, high=70, size=40)

labels = []
for age in ages:
    if age < 30:
        labels.append(0)
    else:
        labels.append(1)        

for i in range(0, 3):
    r = np.random.randint(0, len(labels) - 1)
    if  labels[r] == 0:
        labels[r] = 1
    else:
        labels[r] = 0


#plt.scatter(ages, labels, color='red')
#plt.show()

# Predissão usando regressão linear

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(ages.reshape(-1, 1), labels)

# y = m.x + b

m = model.coef_[0]

b = model.intercept_

# Entendento o Coeficente da reta
''' 
from matplotlib.animation import FuncAnimation

flg, ax = plt.subplots()

axls = plt.axes(xlim = (0, 2),
                ylim = (-0.1, 2))

line, = axls.plot([], [], lw = 3)

def init():
    line.set_data([], [])
    return line,

def animate(i):
    m_copy = i * 0.01
    plt.title('m = ' + str(m_copy))
    x = np.arange(0.0, 10.0, 0.1)
    y = m_copy * x + b
    line.set_data(x, y)

    return line,
'''

#ani = FuncAnimation(flg, animate, init_func= init, frames = 200, interval = 20, blit = True)

#ani.save('m.mp4', writer='ffmpeg', fps=30)

# Regressão Linear daquele conjunto de pontos

# y = m.x + b
'''
linear_idade = (0.5 - b) / m 
print(linear_idade)

plt.plot(ages, ages * m + b, color='blue')
plt.plot([linear_idade, linear_idade], [0, 0.5], '--', color='green')
plt.scatter(ages, labels, color='red')
plt.show()
'''

#Função logistica

import math

def sigmoid(x):
    a = []
    for item in x:
        a.append(1/(1+math.exp(-item)))
    return a

x = np.arange(-10., 10., 0.2)
slg = sigmoid(x)

'''
plt.plot(x, slg)
plt.show()
'''
#Classificador Sigmoide 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(ages.reshape(-1, 1), labels)

#y = m.x + b

m = model.coef_[0][0]
b = model.intercept_[0]

x = np.arange(0, 70, 0.1)

slg = sigmoid(m * x + b)

plt.scatter(ages, labels, color='red')

plt.plot(x, slg)
plt.show()



