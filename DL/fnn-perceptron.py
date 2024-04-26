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
