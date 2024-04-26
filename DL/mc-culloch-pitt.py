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
