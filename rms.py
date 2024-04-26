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


def RMS():
    w, b, lr, epochs = 2, -2, 0.1, 100
    v_w, v_b, eps, beta = 0, 0, 1e-8, 0.9
    errors = []

    print(f"Initial error: {error(w, b, X, Y)}")

    for _ in range(epochs):
        errors.append(error(w, b, X, Y))
        dw, db = 0, 0

        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)

        v_w = beta * v_w + (1 - beta) * dw ** 2
        v_b = beta * v_b + (1 - beta) * db ** 2

        w -= (lr / (np.sqrt(v_w) + eps)) * dw
        b -= (lr / (np.sqrt(v_b) + eps)) * db

    final_error = error(w, b, X, Y)
    print(f"Final error: {final_error}, Weight: {w}, Bias: {b}")

    return errors

list_error = RMS()
plot_errors(list_error)





Journal
import numpy as np

def initialize_weights(input_size, output_size):
    np.random.seed(42)
    return np.random.randn(input_size, output_size) * 0.01

def accelerated_gradient_descent(grad_func, initial_params, learning_rate=0.01, momentum=0.9, num_iterations=100):
    params = initial_params
    velocity = np.zeros_like(initial_params)
    for _ in range(num_iterations):
        grad = grad_func(params - momentum * velocity)
        velocity = momentum * velocity + learning_rate * grad
        params = params - velocity
    return params

def rmsprop(grad_func, initial_params, learning_rate=0.01, decay_rate=0.9, epsilon=1e-8, num_iterations=100):
    params = initial_params
    squared_gradients = np.zeros_like(initial_params)
    for _ in range(num_iterations):
        grad = grad_func(params)
        squared_gradients = decay_rate * squared_gradients + (1 - decay_rate) * grad ** 2
        params = params - learning_rate / (np.sqrt(squared_gradients) + epsilon) * grad
    return params

input_size = 10
output_size = 5
 
weights_initial = initialize_weights(input_size, output_size)

def gradient_function(params):
    return np.random.randn(*params.shape)

params_agd = accelerated_gradient_descent(gradient_function, weights_initial)
print("Parameters after AGD optimization:")
print(params_agd)

params_rmsprop = rmsprop(gradient_function, weights_initial)
print("Parameters after RMSProp optimization:")
print(params_rmsprop)
