import numpy as np

def f(w, x):
    return w @ x


def gradf(w, x):
    return x


def gradient_descent(f, gradf, eta, w, x, iteration=10000):
    prev = 100000
    while abs((f_val := (f(w, x))  - prev)) >= 0.001 or iteration >= 0:
        prev = f_val
        w = w - eta * gradf(w, x)
        iteration -= 1
        
    return w


###########

import matplotlib.pyplot as plt

def gradient_descent(X,y):
    m = b = 0
    iteration = 1000
    learning_rate = 0.001
    n = len(X)

    for i in range(iteration):
        y_pred = m * X + b # slope intercept formula
        cost = (1/n)*sum([val**2 for val in (y-y_pred)]) # Cost function

        # partial derivative with respect to m
        md = -(2/n) * sum(X*(y - y_pred))

        # partial derivative with respect to b
        bd = -(2/n) * sum(y - y_pred)

        # updating m and b values
        m = m - learning_rate * md
        b = b - learning_rate * bd

    print("M:{}, B:{}, iter:{}, cost:{} pred:{}".format(m,b,i,cost, y_pred))

    # plotting the regression line formed
    plt.scatter(X, y)
    plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')
    plt.show()


X = np.array([[3],[2],[4],[5],[9],[6],[2],[8]])
y = np.array([[5],[5],[7],[10],[16],[12],[3],[14]])


Gradient_Descent(X,y)