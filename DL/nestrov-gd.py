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


def nesterov_gradient_descent():
    w, b, lr, epochs = 2, -2, 0.1, 100
    errors = []

    print(f"Initial error: {error(w, b, X, Y)}")
    prev_v_w, prev_v_b, gamma = 0, 0, 0.9

    for _ in range(epochs):
        errors.append(error(w, b, X, Y))
        dw, db = 0, 0

        for x, y in zip(X, Y):
            dw += grad_w(w - gamma * prev_v_w, b - gamma * prev_v_b, x, y)
            db += grad_b(w - gamma * prev_v_w, b - gamma * prev_v_b, x, y)

        v_w = gamma * prev_v_w + lr * dw
        v_b = gamma * prev_v_b + lr * db

        w -= v_w
        b -= v_b

        prev_v_w = v_w
        prev_v_b = v_b

    final_error = error(w, b, X, Y)
    print(f"Final error: {final_error}, Weight: {w}, Bias: {b}")

    return errors


error_list = nesterov_gradient_descent()
plot_errors(error_list)
