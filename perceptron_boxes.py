import numpy as np

# Generate dataset: box number (1-100), label 1 if perfect square, else 0
def is_perfect_square(n):
    return int(np.sqrt(n)) ** 2 == n

data = []
for box in range(1, 101):
    label = 1 if is_perfect_square(box) else 0
    data.append([box, label])
data = np.array(data)

# Normalize box numbers for better training (0-1 scale)
X = data[:, 0:1] / 100.0
y = data[:, 1:2]

# Perceptron model: sigmoid(w*x + b)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

# Initialize weights and bias
np.random.seed(42)
w = np.random.randn(1, 1)
b = np.random.randn(1)

learning_rate = 0.5
max_epochs = 10000

for epoch in range(max_epochs):
    # Forward pass
    z = np.dot(X, w) + b  # shape (100, 1)
    y_pred = sigmoid(z)
    loss = np.mean((y_pred - y) ** 2)

    # Backward pass (gradient descent)
    dz = 2 * (y_pred - y) * sigmoid_deriv(z)
    dw = np.dot(X.T, dz) / len(X)
    db = np.mean(dz)

    w -= learning_rate * dw
    b -= learning_rate * db

    if epoch % 500 == 0 or epoch == max_epochs - 1:
        print(f"Epoch {epoch}: Loss = {loss:.5f}")

# Test and print results
print("\nBox\tPredicted\tActual")
for i in range(1, 101):
    x_norm = i / 100.0
    pred = sigmoid(x_norm * w + b)[0][0]
    print(f"{i}\t{pred:.3f}\t\t{is_perfect_square(i)}")

# Print learned weights and bias
print(f"\nLearned weight: {w[0][0]:.4f}, bias: {b[0]:.4f}") 
