import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.engine import Tensor
import numpy as np
import torch


np.random.seed(42)
torch.manual_seed(42)


def test_add():
    a_np = np.random.randn(3, 4)
    b_np = np.random.randn(3, 4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    (a_pt + b_pt).sum().backward()

    a_tg = Tensor(a_np)
    b_tg = Tensor(b_np)
    (a_tg + b_tg).sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(b_tg.grad, b_pt.grad.numpy(), atol=1e-6)
    print("add: OK")


def test_add_broadcast():
    a_np = np.random.randn(3, 4)
    b_np = np.random.randn(4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    (a_pt + b_pt).sum().backward()

    a_tg = Tensor(a_np)
    b_tg = Tensor(b_np)
    (a_tg + b_tg).sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(b_tg.grad, b_pt.grad.numpy(), atol=1e-6)
    print("add (broadcast): OK")


def test_mul():
    a_np = np.random.randn(3, 4)
    b_np = np.random.randn(3, 4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    (a_pt * b_pt).sum().backward()

    a_tg = Tensor(a_np)
    b_tg = Tensor(b_np)
    (a_tg * b_tg).sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(b_tg.grad, b_pt.grad.numpy(), atol=1e-6)
    print("mul: OK")


def test_matmul():
    x_np = np.random.randn(4, 3)
    w_np = np.random.randn(3, 5)

    x_pt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    w_pt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    (x_pt @ w_pt).sum().backward()

    x_tg = Tensor(x_np)
    w_tg = Tensor(w_np)
    (x_tg @ w_tg).sum().backward()

    assert np.allclose(x_tg.grad, x_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(w_tg.grad, w_pt.grad.numpy(), atol=1e-6)
    print("matmul: OK")


def test_relu():
    a_np = np.random.randn(3, 4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    a_pt.relu().sum().backward()

    a_tg = Tensor(a_np)
    a_tg.relu().sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    print("relu: OK")


def test_log():
    a_np = np.abs(np.random.randn(3, 4)) + 0.1

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    a_pt.log().sum().backward()

    a_tg = Tensor(a_np)
    a_tg.log().sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    print("log: OK")


def test_exp():
    a_np = np.random.randn(3, 4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    a_pt.exp().sum().backward()

    a_tg = Tensor(a_np)
    a_tg.exp().sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    print("exp: OK")


def test_pow():
    a_np = np.random.randn(3, 4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    (a_pt ** 2).sum().backward()

    a_tg = Tensor(a_np)
    (a_tg ** 2).sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    print("pow: OK")


def test_sum_all():
    a_np = np.random.randn(3, 4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    a_pt.sum().backward()

    a_tg = Tensor(a_np)
    a_tg.sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    print("sum (all): OK")


def test_sum_axis():
    a_np = np.random.randn(3, 4)

    a_pt = torch.tensor(a_np, dtype=torch.float64, requires_grad=True)
    a_pt.sum(dim=0).sum().backward()

    a_tg = Tensor(a_np)
    a_tg.sum(axis=0).sum().backward()

    assert np.allclose(a_tg.grad, a_pt.grad.numpy(), atol=1e-6)
    print("sum (axis): OK")


def test_full_mlp_graph():
    x_np = np.random.randn(4, 3)
    w_np = np.random.randn(3, 5)
    b_np = np.random.randn(5)

    x_pt = torch.tensor(x_np, dtype=torch.float64, requires_grad=True)
    w_pt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    (x_pt @ w_pt + b_pt).relu().sum().backward()

    x_tg = Tensor(x_np)
    w_tg = Tensor(w_np)
    b_tg = Tensor(b_np)
    (x_tg @ w_tg + b_tg).relu().sum().backward()

    assert np.allclose(x_tg.grad, x_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(w_tg.grad, w_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(b_tg.grad, b_pt.grad.numpy(), atol=1e-6)
    print("full MLP graph (matmul + add + relu): OK")


def test_conv2d():
    import torch.nn.functional as F
    from core.engine import conv2d_forward_kernel, conv2d_backward_kernel

    np.random.seed(5)
    inp_np = np.random.randn(1, 1, 8, 8)
    w_np = np.random.randn(2, 1, 3, 3)
    b_np = np.random.randn(2)

    inp_pt = torch.tensor(inp_np, dtype=torch.float64, requires_grad=True)
    w_pt = torch.tensor(w_np, dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_np, dtype=torch.float64, requires_grad=True)
    out_pt = F.conv2d(inp_pt, w_pt, b_pt)
    out_pt.sum().backward()

    from nn import Conv2d
    conv = Conv2d(1, 2, 3)
    conv.weight = Tensor(w_np)
    conv.bias = Tensor(b_np)
    out_tg = conv(Tensor(inp_np))
    out_tg.sum().backward()

    assert np.allclose(out_tg.data, out_pt.detach().numpy(),
                       atol=1e-6), "forward mismatch"
    assert np.allclose(conv.weight.grad, w_pt.grad.numpy(),
                       atol=1e-6), "weight grad mismatch"
    assert np.allclose(conv.bias.grad, b_pt.grad.numpy(),
                       atol=1e-6), "bias grad mismatch"
    print("conv2d (forward + backward): OK")


if __name__ == '__main__':
    print("Running parity tests against PyTorch...\n")
    test_add()
    test_add_broadcast()
    test_mul()
    test_matmul()
    test_relu()
    test_log()
    test_exp()
    test_pow()
    test_sum_all()
    test_sum_axis()
    test_full_mlp_graph()
    test_conv2d()
    print("\nAll parity tests passed. TitanGrad matches PyTorch.")