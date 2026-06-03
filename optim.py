from core.backend import get_backend


class SGD:
    def __init__(self, parameters, lr=0.01):
        self.params = parameters
        self.lr = lr

    def step(self):
        for p in self.params:
            if p.requires_grad:
                p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                xp = get_backend(p.device)
                p.grad = xp.zeros_like(p.data)


class Adam:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = parameters
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0  # time step
        self.m = {}  # first moment aka momentum
        self.v = {}  # second moment aka velo

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue

            xp = get_backend(p.device)

            if i not in self.m:
                self.m[i] = xp.zeros_like(p.data)
                self.v[i] = xp.zeros_like(p.data)

            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
