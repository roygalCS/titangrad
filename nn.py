import numpy as np
from engine import Tensor


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
        self.w = Tensor(np.random.randn(nin, 1) * 0.1)
        self.b = Tensor(0.0)
        self.activation = activation

    def __call__(self, x):
        if isinstance(x, list):
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
