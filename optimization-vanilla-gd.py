import numpy as np
import matplotlib.pyplot as plt

X = [0.3, 0.5, 0.7, 0.4]
Y = [0.2, 0.6, 0.9, 0.3]

def sigmoid(w, x, b):
    return 1.0 / (1.0 + np.exp(-(w * x + b)))

def error(w, b, X, Y):
    total_error = 0.0
    for x, y in zip(X, Y):
        y_pred = sigmoid(w, x, b)
        total_error += 0.5 * (y_pred - y) ** 2
    return total_error

def grad_w(w, b, X, Y):
    y_pred = sigmoid(w, X, b)
    return (y_pred - Y) * y_pred * (1 - y_pred) * X

def grad_b(w, b, X, Y):
    y_pred = sigmoid(w, X, b)
    return (y_pred - Y) * y_pred * (1 - y_pred)

def plot_errors(errors):
    epochs = range(len(errors))
    plt.plot(epochs, errors)
    plt.xlabel("Epoch")
    plt.ylabel("Error")
    plt.title("Error vs. Epoch")
    plt.show()



def gradient_descent():
    w, b, lr, epochs = 2, -2, 1, 100
    errors = []

    print(f"Initial error: {error(w, b, X, Y)}")

    for _ in range(epochs):
        errors.append(error(w, b, X, Y))
        dw, db = 0, 0

        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        w -= lr * dw
        b -= lr * db

    final_error = error(w, b, X, Y)
    print(f"Final error: {final_error}, Weight: {w}, Bias: {b}")

    return errors

error_list = gradient_descent()
plot_errors(error_list)
