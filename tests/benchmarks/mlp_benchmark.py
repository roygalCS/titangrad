"""
Benchmarks a full MLP forward+backward pass on the NumPy backend vs. the
CuPy backend, both obtained via core.backend.get_backend().

There's no existing matmul/MLP entry in tests/benchmarks/ (only
conv_benchmark.py, which times conv2d), so the op graph and dtype here
mirror what's actually exercised elsewhere in the repo:
  - op graph (matmul -> +bias -> relu, chained) matches
    tests/test_parity.py::test_full_mlp_graph
  - dims (784 -> 256 -> 128 -> 10) match the MNIST scale used in
    examples/mnist_classifier.py
  - dtype (float64) matches core.engine.Tensor, which always stores
    float64 data

Batch size (1024) is chosen to be large enough that per-call/kernel-launch
overhead doesn't dominate the timing, which the tiny shapes in
test_parity.py (e.g. 4x3) are too small for.

This benchmarks raw array ops via get_backend() directly (not
core.engine.Tensor's autograd graph), so it works identically on both
backends without touching any existing code.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from core.backend import get_backend

BATCH = 1024
DIMS = [784, 256, 128, 10]
N_ITERS = 100
N_WARMUP = 5


def build_params(xp, seed=0):
    xp.random.seed(seed)
    params = []
    for nin, nout in zip(DIMS[:-1], DIMS[1:]):
        scale = (2.0 / nin) ** 0.5
        W = (xp.random.randn(nin, nout) * scale).astype(xp.float64)
        b = xp.zeros(nout, dtype=xp.float64)
        params.append((W, b))
    return params


def mlp_forward_backward(xp, X, params):
    """matmul -> +bias -> relu, chained across layers (linear on the last
    layer), then the backward pass for L = sum(out)."""
    activations = [X]
    zs = []
    a = X
    for i, (W, b) in enumerate(params):
        z = a @ W + b
        zs.append(z)
        a = xp.maximum(z, 0) if i < len(params) - 1 else z
        activations.append(a)

    grad = xp.ones_like(activations[-1])
    grads = []
    for i in reversed(range(len(params))):
        W, _ = params[i]
        z = zs[i]
        if i < len(params) - 1:
            grad = grad * (z > 0)
        a_prev = activations[i]
        dW = a_prev.T @ grad
        db = grad.sum(axis=0)
        grads.append((dW, db))
        grad = grad @ W.T
    return grads


def time_backend(xp, is_gpu):
    X = (xp.random.randn(BATCH, DIMS[0])).astype(xp.float64)
    params = build_params(xp)

    for _ in range(N_WARMUP):
        mlp_forward_backward(xp, X, params)
    if is_gpu:
        xp.cuda.Device().synchronize()

    start = time.perf_counter()
    for _ in range(N_ITERS):
        mlp_forward_backward(xp, X, params)
    if is_gpu:
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start

    return elapsed / N_ITERS


def main():
    print(f"Op: full MLP forward+backward   dims={DIMS}   batch={BATCH}")
    print(f"Iterations: {N_ITERS} (+{N_WARMUP} warmup, discarded)\n")

    numpy_mean = time_backend(get_backend('cpu'), is_gpu=False)
    print(f"NumPy mean time: {numpy_mean * 1000:.3f} ms/iter")

    try:
        cp = get_backend('gpu')
    except RuntimeError as e:
        print(f"\nCuPy backend not available: {e}")
        print("Skipping GPU benchmark and speedup ratio.")
        return

    cupy_mean = time_backend(cp, is_gpu=True)
    print(f"CuPy mean time:  {cupy_mean * 1000:.3f} ms/iter")
    print(f"\nSpeedup (NumPy/CuPy): {numpy_mean / cupy_mean:.2f}x")


if __name__ == '__main__':
    main()
