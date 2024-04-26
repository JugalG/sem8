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


def adam():
    w, b, lr, epochs = 2, -2, 0.1, 100
    v_w, v_b, eps, beta = 0, 0, 1e-8, 0.9
    m_w, m_b = 0, 0
    beta2 = 0.999
    errors = []

    print(f"Initial error: {error(w, b, X, Y)}")

    for i in range(epochs):
        errors.append(error(w, b, X, Y))
        dw, db = 0, 0

        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        m_w = beta * m_w + (1 - beta) * dw
        m_b = beta * m_b + (1 - beta) * db

        v_w = beta2 * v_w + (1 - beta2) * dw ** 2
        v_b = beta2 * v_b + (1 - beta2) * db ** 2

        m_w_hat = m_w / (1 - beta ** (i + 1))
        m_b_hat = m_b / (1 - beta ** (i + 1))

        v_w_hat = v_w / (1 - beta2 ** (i + 1))
        v_b_hat = v_b / (1 - beta2 ** (i + 1))

        w -= (lr / (np.sqrt(v_w_hat) + eps)) * m_w_hat
        b -= (lr / (np.sqrt(v_b_hat) + eps)) * m_b_hat

    final_error = error(w, b, X, Y)
    print(f"Final error: {final_error}, Weight: {w}, Bias: {b}")

    return errors

list_error = adam()
plot_errors(list_error)
