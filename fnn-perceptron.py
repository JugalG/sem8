import numpy as np

# Input features and corresponding labels
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 0, 0, 1])

# Learning rate
learning_rate = 0.1

# Random initialization of weights
weights = np.random.randn(X.shape[1])
print("Initial weights:", weights)

# Training loop
for epoch in range(5):  # Number of epochs
    errors = 0
    print("Epoch:", epoch + 1)
    print("Initial weights:", weights)

    # Iterate through each training example
    for x, y_true in zip(X, Y):
        # Predict the output
        y_pred = np.dot(x, weights)

        # Convert prediction to binary output
        y_pred_binary = 1 if y_pred >= 0 else 0

        # Calculate the error
        error = y_true - y_pred_binary

        if error != 0:
            print(f"Error: {error}, Weight updated, Input: {x}, Label: {y_true}")
            errors += 1
            weights += learning_rate * error * x

        print("Final weights:", weights)

    # Check if all examples are classified correctly
    if errors == 0:
        print("Training complete.")
        break



Journal

import numpy as np

# Define the activation function (step function)
def step_function(x):
    return 1 if x > 0 else 0

# Define the perceptron function
def perceptron(input_data, weights, bias):
    activation = np.dot(input_data, weights) + bias
    return step_function(activation)

# Define the truth tables for AND, OR, and NOT gates
gate_inputs = {
    'AND': np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]),
    'OR': np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1]
    ]),
    'NOT': np.array([
        [0],
        [1]
    ])
}

gate_outputs = {
    'AND': np.array([0, 0, 0, 0, 0, 0, 0, 1]),
    'OR': np.array([0, 1, 1, 1, 1, 1, 1, 1]),
    'NOT': np.array([1, 0])
}

# Initialize weights and bias randomly for each gate
gate_weights = {
    'AND': np.random.rand(3),
    'OR': np.random.rand(3),
    'NOT': np.random.rand(1)
}

gate_bias = {
    'AND': np.random.rand(1),
    'OR': np.random.rand(1),
    'NOT': np.random.rand(1)
}

# Training the perceptrons for each gate
learning_rate = 0.1
epochs = 100

for gate in gate_inputs:
    for epoch in range(epochs):
        for i in range(len(gate_inputs[gate])):
            prediction = perceptron(gate_inputs[gate][i], gate_weights[gate], gate_bias[gate])
            error = gate_outputs[gate][i] - prediction
            gate_weights[gate] += learning_rate * error * gate_inputs[gate][i]
            gate_bias[gate] += learning_rate * error

# Test the trained perceptrons
for gate in gate_inputs:
    print("Trained weights for", gate, "gate:", gate_weights[gate])
    print("Trained bias for", gate, "gate:", gate_bias[gate])

# Test the perceptrons with all possible inputs for each gate
for gate in gate_inputs:
    print("Testing the", gate, "gate:")
    for i in range(len(gate_inputs[gate])):
        prediction = perceptron(gate_inputs[gate][i], gate_weights[gate], gate_bias[gate])
        print("Input:", gate_inputs[gate][i], "Predicted Output:", prediction)
