import numpy as np
from sklearn.datasets import fetch_openml
from nn import MLP


def softmax_cross_entropy(logits, target_index):
    """
    logits: a Tensor of shape (num_classes), outputs from network
    target_index: an integer, which class is correct
    """

    max_val = max(float(l.data) for l in logits)
    shifted = [l - max_val for l in logits]
    exp_vals = [l.exp() for l in shifted]

    sum_exp = exp_vals[0]
    for e in exp_vals[1:]:
        sum_exp = sum_exp + e

    correct_prob = exp_vals[target_index] / sum_exp
    loss = -correct_prob.log()

    return loss


print("Loading MNIST dataset from OpenML...")
# as_frame=False guarantees NumPy arrays instead of Pandas DataFrames(imported by sklearn)
mnist = fetch_openml('mnist_784', version=1, as_frame=False)

# Normalize pixel values from [0, 255] down to [0.0, 1.0]
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

# Split data into 60,000 training images and 10,000 testing images(due to 70,000 total)
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


model = MLP(784, [128, 64, 10])
learning_rate = 0.01

print("Starting Training Process...")

# epoch is one complete pass thru dataset training, in case of 60k -> 600k
for epoch in range(10):
    total_loss = 0.0
    correct = 0

    for image, label in zip(X_train, y_train):

        logits = model(image)
        loss = softmax_cross_entropy(logits, label)
        model.zero_grad()
        loss.backward()

        for p in model.parameters():
            p.data -= learning_rate * p.grad

        total_loss += float(loss.data)

        pred = max(range(10), key=lambda i: float(logits[i].data))
        if pred == label:
            correct += 1

    accuracy = (correct / len(y_train)) * 100
    avg_loss = total_loss / len(y_train)
    print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
