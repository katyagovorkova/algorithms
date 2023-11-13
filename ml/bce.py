import pandas as pd
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

from softmax import softmax

def fit(X,y, c, epochs, learn_rate):

    # Splitting the number of training examples and features
    (m,n) = X.shape

    # Selecting random weights and bias
    w = np.random.random((n,c))
    b = np.random.random(c)

    loss_arr = []

    # Training
    for epoch in range(epochs):

        # Hypothesis function
        z = X@w + b

        # Computing gradient of loss w.r.t w and b
        grad_for_w = (1/m)*np.dot(X.T,Softmax(z) - OneHot(y, c))
        grad_for_b = (1/m)*np.sum(Softmax(z) - OneHot(y, c))

        # Updating w and b
        w = w - learn_rate * grad_for_w
        b = b - learn_rate * grad_for_b

        # Computing the loss
        loss = -np.mean(np.log(Softmax(z)[np.arange(len(y)), y]))
        loss_arr.append(loss)
        print("Epoch: {} , Loss: {}".format(epoch, loss))

    return w, b, loss_arr

def predict(X, w, b):

    z = X@w + b
    y_hat = Softmax(z)

    # Returning highest probability class.
    return np.argmax(y_hat, axis=1)

def OneHot(y, c):

  # Constructing zero matrix
    y_encoded = np.zeros((len(y), c))

  # Giving 1 for some colums
    y_encoded[np.arange(len(y)), y] = 1
    return y_encoded

# Splitting the dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])

# Reshaping the data into 28*28 equal matrix
X_train = X_train.reshape(60000,28*28)
X_test = X_test.reshape(10000,28*28)

# Normalizing the training set
X_train = X_train/300

# Training the model
w, b, loss = fit(X_train, y_train, c=10, epochs=1000, learn_rate=0.10)

predictions = predict(X_train, w, b)
actual_values = y_train

print("Predictions:", predictions)
print("Actual values:", actual_values)

accuracy = np.sum(actual_values==predictions)/len(actual_values)
print(accuracy)