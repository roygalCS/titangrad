from numba import jit
import numpy as np
from core.backend import get_backend


@jit(nopython=True)
def conv2d_forward_kernel(input_data, kernel, stride=1, padding=0):
    batch, in_c, h, w = input_data.shape
    out_c, _, kh, kw = kernel.shape

    out_h = (h + 2*padding - kh) // stride + 1
    out_w = (w + 2*padding - kw) // stride + 1

    output = np.zeros((batch, out_c, out_h, out_w))

    for b in range(batch):
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    for ic in range(in_c):
                        for khi in range(kh):
                            for kwi in range(kw):
                                ih = oh * stride + khi - padding
                                iw = ow * stride + kwi - padding
                                if 0 <= ih < h and 0 <= iw < w:
                                    output[b, oc, oh, ow] += \
                                        input_data[b, ic, ih, iw] * \
                                        kernel[oc, ic, khi, kwi]
    return output


@jit(nopython=True)
def conv2d_backward_kernel(d_out, input_data, kernel, stride=1, padding=0):
    batch, in_c, h, w = input_data.shape
    out_c, _, kh, kw = kernel.shape
    _, _, out_h, out_w = d_out.shape

    d_input = np.zeros_like(input_data)
    d_kernel = np.zeros_like(kernel)

    for b in range(batch):
        for oc in range(out_c):
            for oh in range(out_h):
                for ow in range(out_w):
                    grad = d_out[b, oc, oh, ow]
                    for ic in range(in_c):
                        for khi in range(kh):
                            for kwi in range(kw):
                                ih = oh * stride + khi - padding
                                iw = ow * stride + kwi - padding
                                if 0 <= ih < h and 0 <= iw < w:
                                    d_input[b, ic, ih, iw] += grad * \
                                        kernel[oc, ic, khi, kwi]
                                    d_kernel[oc, ic, khi, kwi] += grad * \
                                        input_data[b, ic, ih, iw]

    return d_input, d_kernel


def _unbroadcast(grad, target_shape):

    # scalar case:

    if target_shape == ():
        return grad.sum()

    while len(grad.shape) > len(target_shape):
        grad = grad.sum(axis=0)

    for i, dim in enumerate(target_shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:

    def __init__(self, data, _children=(), _op='', device='cpu', requires_grad=True):
        self.device = device
        self.requires_grad = requires_grad

        xp = get_backend(device)

        self.data = xp.asarray(data, dtype=np.float64)
        self.grad = xp.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def to(self, device):
        """ pytorch copy of .to('cuda') """

        if device == self.device:
            return self
        xp = get_backend(device)
        new_data = xp.asarray(self.data)
        return Tensor(new_data, device=device, requires_grad=self.requires_grad)

    def __repr__(self):
        return f"Tensor(data={self.data}, shape={self.data.shape})"

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __getitem__(self, idx):
        out = Tensor(self.data[idx], (self,), 'slice')

        def _backward():
            grad_update = np.zeros_like(self.data)
            grad_update[idx] = out.grad
            self.grad += grad_update

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += _unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            if out.data.ndim == 0:
                # Case 1: 1D vector dot product → scalar result
                # dL/dA = dL/dC * B,  dL/dB = dL/dC * A
                self.grad += out.grad * other.data
                other.grad += out.grad * self.data
            else:
                # Case 2: normal matrix multiply
                self.grad += out.grad @ other.data.T
                other.grad += self.data.T @ out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(
            axis=axis, keepdims=keepdims), (self,), 'sum')

        def _backward():
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)
        out._backward = _backward
        return out

    def exp(self):
        out = Tensor(np.exp(self.data), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    def log(self):
        out = Tensor(np.log(self.data), (self, ), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * (other**-1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)
                          ), "for now only supporting int and float exponent"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward

        return out

    def tanh(self):

        t = np.tanh(self.data)
        out = Tensor(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()
