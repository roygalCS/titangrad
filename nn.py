import numpy as np
from core.engine import Tensor, conv2d_forward_kernel, conv2d_backward_kernel


class Module:
    def zero_grad(self):
        for p in self.parameters():
            if isinstance(p.grad, np.ndarray):
                p.grad.fill(0.0)
            else:
                p.grad = 0.0

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, activation='tanh'):
        self.w = Tensor(np.random.randn(nin) * 0.1)
        self.b = Tensor(0.0)
        self.activation = activation

    def __call__(self, x):
        if isinstance(x, list):
            if len(x) > 0 and isinstance(x[0], Tensor):
                x = Tensor(np.array([float(t.data)
                           for t in x], dtype=np.float64))
            else:
                x = Tensor(np.array(x, dtype=np.float64))
        elif not isinstance(x, Tensor):
            x = Tensor(x)
        act = x @ self.w + self.b

        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        else:
            return act

    def parameters(self):
        return [self.w, self.b]


class Layer(Module):
    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):
    def __init__(self, nin, nouts, activation='tanh'):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1],
                  activation=activation if i < len(nouts)-1 else 'linear')
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Tensor(np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size) * scale)
        self.bias = Tensor(np.zeros(out_channels))

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)

        out_data = conv2d_forward_kernel(
            x.data, self.weight.data, self.stride, self.padding)
        out_data = out_data + \
            self.bias.data[np.newaxis, :, np.newaxis, np.newaxis]

        out = Tensor(out_data, (x, self.weight, self.bias), 'conv2d')

        input_snapshot = x.data.copy()
        weight_snapshot = self.weight.data.copy()

        def _backward():
            d_input, d_weight = conv2d_backward_kernel(
                out.grad, input_snapshot, weight_snapshot, self.stride, self.padding
            )
            x.grad += d_input
            self.weight.grad += d_weight
            self.bias.grad += out.grad.sum(axis=(0, 2, 3))

        out._backward = _backward
        return out

    def parameters(self):
        return [self.weight, self.bias]


class Flatten(Module):
    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x)
        batch = x.data.shape[0]
        out = Tensor(x.data.reshape(batch, -1), (x,), 'flatten')

        def _backward():
            x.grad += out.grad.reshape(x.data.shape)

        out._backward = _backward
        return out

    def parameters(self):
        return []


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
