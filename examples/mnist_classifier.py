import numpy as np
import urllib.request
import gzip
import os
from core.engine import Tensor
from nn import MLP
from optim import Adam


def load_mnist():
    base_url = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images':  't10k-images-idx3-ubyte.gz',
        'test_labels':  't10k-labels-idx1-ubyte.gz',
    }
    cache_dir = os.path.expanduser('~/.cache/mnist')
    os.makedirs(cache_dir, exist_ok=True)

    def fetch(filename):
        path = os.path.join(cache_dir, filename)
        if not os.path.exists(path):
            print(f"  Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, path)
        return path

    def read_images(path):
        with gzip.open(path, 'rb') as f:
            f.read(16)
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 784)

    def read_labels(path):
        with gzip.open(path, 'rb') as f:
            f.read(8)
            return np.frombuffer(f.read(), dtype=np.uint8).astype(int)

    X_train = read_images(fetch(files['train_images'])).astype(
        np.float32) / 255.0
    y_train = read_labels(fetch(files['train_labels']))
    X_test = read_images(fetch(files['test_images'])).astype(
        np.float32) / 255.0
    y_test = read_labels(fetch(files['test_labels']))
    return X_train, y_train, X_test, y_test


print("Loading MNIST...")
X_train, y_train, X_test, y_test = load_mnist()

# Build model
model = MLP(784, [128, 64, 10], activation='relu')
optimizer = Adam(model.parameters(), lr=0.001)


def cross_entropy_loss(logits_list, target):
    """logits_list: list of 10 Tensors, target: integer"""
    max_val = float(max(float(l.data) for l in logits_list))
    shifted = [l - max_val for l in logits_list]
    exp_vals = [l.exp() for l in shifted]

    sum_exp = exp_vals[0]
    for e in exp_vals[1:]:
        sum_exp = sum_exp + e

    log_sum_exp = sum_exp.log()
    log_probs = [s - log_sum_exp for s in shifted]
    loss = -log_probs[target]
    return loss


print("Training...")
for epoch in range(5):
    total_loss = 0.0
    correct = 0

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
