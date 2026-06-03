import numpy as np
from core.engine import Tensor
from nn import MLP
from optim import Adam

from sklearn.datasets import fetch_openml
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype(np.float32) / 255.0
y = mnist.target.astype(int)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# Build model
model = MLP(784, [128, 64, 10], activation='relu')
optimizer = Adam(model.parameters(), lr=0.001)


def cross_entropy_loss(logits_list, target):
    """logits_list: list of 10 Tensors, target: integer"""
    # Stack logits into one Tensor
    data = np.array([float(l.data) for l in logits_list])
    logits = Tensor(data)

    # Numerically stable softmax + cross-entropy
    shifted = logits - logits.data.max()
    exp_vals = shifted.exp()
    log_sum_exp = exp_vals.sum().log()
    log_prob = shifted - log_sum_exp  # log probabilities
    # negative log likelihood of correct class
    loss = -log_prob[target]
    return loss


print("Training...")
for epoch in range(5):
    total_loss = 0.0
    correct = 0

    # Mini-batch SGD (batch_size=32)
    indices = np.random.permutation(len(X_train))

    for start in range(0, min(len(X_train), 10000), 1):  # 10k examples per epoch for speed
        i = indices[start]

        optimizer.zero_grad()
        logits = model(X_train[i])
        loss = cross_entropy_loss(logits, y_train[i])
        loss.backward()
        optimizer.step()

        total_loss += float(loss.data)
        pred = max(range(10), key=lambda j: float(logits[j].data))
        if pred == y_train[i]:
            correct += 1

    print(
        f"Epoch {epoch+1}: loss={total_loss/10000:.4f}, train_acc={correct/10000*100:.1f}%")
