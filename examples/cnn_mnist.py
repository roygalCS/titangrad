import sys
import os
import gzip
import urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from optim import Adam
from nn import Conv2d, MLP, Flatten, Sequential
from core.engine import Tensor
import numpy as np


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


def cross_entropy_loss(logits_list, target):
    max_val = float(max(float(l.data) for l in logits_list))
    shifted = [l - max_val for l in logits_list]
    exp_vals = [l.exp() for l in shifted]
    sum_exp = exp_vals[0]
    for e in exp_vals[1:]:
        sum_exp = sum_exp + e
    log_sum_exp = sum_exp.log()
    log_probs = [s - log_sum_exp for s in shifted]
    return -log_probs[target]


if __name__ == '__main__':
    print("Loading MNIST...")
    X_train, y_train, X_test, y_test = load_mnist()

    # Reshape flat 784 vectors into (1, 28, 28) images for the conv layer
    X_train_img = X_train.reshape(-1, 1, 28, 28).astype(np.float64)
    X_test_img = X_test.reshape(-1, 1, 28, 28).astype(np.float64)

    # Architecture:
    # Conv2d(1->8, 3x3) -> ReLU -> flatten (8*26*26=5408) -> Linear(5408->64) -> Linear(64->10)
    conv = Conv2d(1, 8, kernel_size=3, stride=1, padding=0)
    mlp = MLP(8 * 26 * 26, [64, 10], activation='relu')
    flat = Flatten()

    optimizer = Adam(conv.parameters() + mlp.parameters(), lr=0.001)

    def forward(image):
        x = conv(image)
        relu_out = x.relu()
        flat_out = flat(relu_out)
        logits = mlp(flat_out)
        return logits

    print("Training CNN...")
    for epoch in range(5):
        total_loss = 0.0
        correct = 0

        indices = np.random.permutation(len(X_train_img))

        for start in range(0, min(len(X_train_img), 2000), 1):
            i = indices[start]

            optimizer.zero_grad()
            logits = forward(X_train_img[i:i+1])
            loss = cross_entropy_loss(logits, y_train[i])
            loss.backward()
            optimizer.step()

            total_loss += float(loss.data)
            pred = max(range(10), key=lambda j: float(logits[j].data))
            if pred == y_train[i]:
                correct += 1

        print(
            f"Epoch {epoch+1}: loss={total_loss/2000:.4f}, train_acc={correct/2000*100:.1f}%")