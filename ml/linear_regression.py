import numpy as np
import matplotlib.pyplot as plt

def mse(X, w, y):
    return np.mean((X@w - y)**2)

def mse_grad(X, w, y):
    return 2 * X.T @ (X @ w - y) / N

def w_from_psudoinverse(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y_true

class LinearRegression():
    def __init__(self, w=np.zeros(2), num_iter=10000):
        self.w = w
        self.num_iter = num_iter

    def gradient_descent(self, X, y, eta):
        prev_f = 100000
        tol = 1e-3
        while abs((f_val := (mse(X, self.w, y)) - prev_f)) >= tol and self.num_iter >= 0:
            prev_f = f_val
            self.w = self.w - eta * mse_grad(X, self.w, y)
            self.num_iter -= 1

        return self.w

N = 1000
w0 = np.array([1, 2])
X = np.empty((N, 2))
X[:, 0] = np.random.normal(0, 1, N)
X[:, 1] = 1
y = X @ w0 + np.random.normal(0, 0.1, N)

plt.plot(X[:, 0], y, ".")

lr = LinearRegression()
print(lr.gradient_descent(X, y, 0.1))
