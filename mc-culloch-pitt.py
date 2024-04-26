import numpy as np

# Input vectors
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]

# Number of elements in the input vectors
n = len(x1)

# Threshold for decision
threshold = float(input("Enter the threshold: "))

# Randomly initialized weights
weights1 = np.random.random(n)
weights2 = np.random.random(n)

print("Weights:", weights1, weights2)

# Calculate the weighted sum
y_sum = 0

# Normal logic
for i in range(n):
    y_sum += (x1[i] * weights1[i] + x2[i] * weights2[i])

print("Weighted sum (Normal):", y_sum)

# Check if weighted sum exceeds threshold (Normal)
if threshold < y_sum:
    print("Accepted")
else:
    print("Failed")

# Calculate the weighted sum (AND logic)
y_sum_and = 0
for i in range(n):
    if x1[i] == 1 and x2[i] == 1:
        y_sum_and += (x1[i] * weights1[i] + x2[i] * weights2[i])

print("Weighted sum (AND):", y_sum_and)

# Check if weighted sum exceeds threshold (AND)
if threshold < y_sum_and:
    print("Accepted")
else:
    print("Failed")

# Calculate the weighted sum (OR logic)
y_sum_or = 0
for i in range(n):
    if x1[i] == 1 or x2[i] == 1:
        y_sum_or += (x1[i] * weights1[i] + x2[i] * weights2[i])

print("Weighted sum (OR):", y_sum_or)

# Check if weighted sum exceeds threshold (OR)
if threshold < y_sum_or:
    print("Accepted")
else:
    print("Failed")







Journal 
# Define the McCulloch-Pitts neuron function
def mcculloch_pitts_neuron(inputs, weights, threshold): 
    activation = sum([x * w for x, w in zip(inputs, weights)]) 
    return 1 if activation >= threshold else 0

# Define the AND gate with two inputs 
def AND_gate(inputs):
    weights = [1, 1] 
    # Weight of 1 for both inputs 
    threshold = 2  # Threshold is 2 for AND gate
    return mcculloch_pitts_neuron(inputs, weights, threshold)

# Define the OR gate with two inputs 
def OR_gate(inputs):
    weights = [1, 1] 
    # Weight of 1 for both inputs 
    threshold = 1  # Threshold is 1 for OR gate
    return mcculloch_pitts_neuron(inputs, weights, threshold)
 
# Define the NOT gate with two inputs 
def NOT_gate(inputs):
    weights = [-1, 0] 
    # Weight of -1 for the input and 0 for the bias 
    threshold = 0  # Threshold is 0 for NOT gate
    return mcculloch_pitts_neuron(inputs, weights, threshold)

# Function to train gates and print weights after each iteration 
def train_gate(gate_func):
    print("Training", gate_func.__name__, "gate:")
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    weights = [1, 1]  # Initial weights
    threshold = 2 if gate_func == AND_gate else 1  # Threshold for different gates
    learning_rate = 0.1

    for epoch in range(5):  # 5 iterations
        print("Weights after iteration", epoch + 1, ":")
        for weight in weights:
            print("[", weight, "]")
        print()

        for i in range(len(inputs)):
            prediction = mcculloch_pitts_neuron(inputs[i], weights, threshold)
            error = prediction - gate_func(inputs[i])  # Calculate error
            for j in range(len(weights)):
                weights[j] -= learning_rate * error * inputs[i][j]  # Update weights

# Train and test gates 
train_gate(AND_gate)
print()
train_gate(OR_gate)
print()
train_gate(NOT_gate)
