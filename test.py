import numpy as np
import torch

from core.engine import Tensor

# Seed everything so both frameworks use same numbers
np.random.seed(42)
torch.manual_seed(42)

# Create identical inputs
x_np = np.random.randn(4, 3)
w_np = np.random.randn(3, 5)
b_np = np.random.randn(5)

# PyTorch
x_torch = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
w_torch = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
b_torch = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)

# Forward pass
out_torch = (x_torch @ w_torch + b_torch).relu()

# Reduce to scalar loss
loss_torch = out_torch.sum()

# Backward pass
loss_torch.backward()

# TitanGrad
x = Tensor(x_np)
w = Tensor(w_np)
b = Tensor(b_np)

# Forward pass
out = (x @ w + b).relu()

# Reduce to scalar loss
loss = out.sum()

# Backward pass
loss.backward()

# Compare gradients
assert np.allclose(x.grad, x_torch.grad.numpy(), atol=1e-6)
assert np.allclose(w.grad, w_torch.grad.numpy(), atol=1e-6)
assert np.allclose(b.grad, b_torch.grad.numpy(), atol=1e-6)

print("Gradient parity achieved.")
print("TitanGrad matches PyTorch.")
