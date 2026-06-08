# TitanGrad

A NumPy-based autograd engine with a PyTorch-like API. Supports tensor operations, automatic differentiation, a neural network library, and a Numba-accelerated Conv2d layer.

## Features

- Reverse-mode autodiff over dynamically built computation graphs
- Broadcasting-aware gradients for add, mul, and matmul
- GPU support via CuPy (`.to('cuda')`)
- Numba-JIT-compiled Conv2d forward and backward kernels
- `nn` module: `Neuron`, `Layer`, `MLP`, `Conv2d`, `Flatten`, `Sequential`
- `optim` module: `SGD`, `Adam`

## Installation

```bash
git clone https://github.com/roygalCS/titangrad.git
cd titangrad
pip install numpy numba
# Optional: pip install torch        (for running parity tests)
# Optional: pip install cupy-cuda12x (for GPU support)
```

## Quick Start

```python
from core.engine import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
w = Tensor([[0.5], [-0.5]])

out = (x @ w).sum()
out.backward()

print(x.grad)   # dL/dx
print(w.grad)   # dL/dw
```

## Neural Network Example

```python
import numpy as np
from core.engine import Tensor
from nn import MLP
from optim import SGD

model = MLP(nin=2, nouts=[4, 4, 1])
optimizer = SGD(model.parameters(), lr=0.01)

X = Tensor(np.random.randn(8, 2))
y = Tensor(np.random.randn(8, 1))

for step in range(100):
    pred = model(X)
    loss = ((pred - y) ** 2).sum()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 20 == 0:
        print(f"step {step}  loss {loss.data:.4f}")
```

## Conv2d Example

```python
import numpy as np
from core.engine import Tensor
from nn import Conv2d, Flatten, Sequential, Layer

model = Sequential(
    Conv2d(1, 8, kernel_size=3, padding=1),
    Flatten(),
    # add Linear layers as needed
)

x = Tensor(np.random.randn(4, 1, 8, 8))   # batch=4, 1 channel, 8x8
out = model(x)
```

## Running Tests

Parity tests compare TitanGrad gradients against PyTorch across all core ops:

```bash
python test.py
```

Expected output:
```
Running parity tests against PyTorch...

add: OK
add (broadcast): OK
mul: OK
matmul: OK
relu: OK
log: OK
exp: OK
pow: OK
sum (all): OK
sum (axis): OK
full MLP graph (matmul + add + relu): OK
conv2d (forward + backward): OK

All parity tests passed. TitanGrad matches PyTorch.
```

## Project Structure

```
titangrad/
├── core/
│   ├── __init__.py
│   ├── engine.py       # Tensor class + autograd engine + Conv2d kernels
│   └── backend.py      # NumPy / CuPy backend selector
├── nn.py               # Module, Neuron, Layer, MLP, Conv2d, Flatten, Sequential
├── optim.py            # SGD, Adam
├── test.py             # Parity tests vs PyTorch
└── Utils.py
```

## Supported Operations

| Operation | Forward | Backward |
|-----------|---------|----------|
| `+` | ✓ | ✓ broadcast-aware |
| `*` | ✓ | ✓ broadcast-aware |
| `@` (matmul) | ✓ | ✓ |
| `**` (pow) | ✓ | ✓ |
| `sum` | ✓ | ✓ axis/keepdims |
| `exp` | ✓ | ✓ |
| `log` | ✓ | ✓ |
| `relu` | ✓ | ✓ |
| `tanh` | ✓ | ✓ |
| `conv2d` | ✓ Numba JIT | ✓ Numba JIT |
| slice `[]` | ✓ | ✓ |

## Device Support

```python
t = Tensor(data)
t_gpu = t.to('cuda')   # requires cupy-cuda12x
t_cpu = t_gpu.to('cpu')
```
