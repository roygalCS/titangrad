from core.engine import conv2d_forward_kernel
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def conv2d_pure_python(input_data, kernel, stride=1, padding=0):
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
                                    output[b, oc, oh, ow] += input_data[b,
                                                                        ic, ih, iw] * kernel[oc, ic, khi, kwi]
    return output


if __name__ == '__main__':
    inp = np.random.randn(1, 1, 28, 28)
    ker = np.random.randn(8, 1, 3, 3)

    print("Warming up Numba (first call compiles the kernel)...")
    conv2d_forward_kernel(inp, ker)
    print("Done.\n")

    N = 20
    t0 = time.perf_counter()
    for _ in range(N):
        conv2d_forward_kernel(inp, ker)
    numba_ms = (time.perf_counter() - t0) / N * 1000

    N_py = 3
    t0 = time.perf_counter()
    for _ in range(N_py):
        conv2d_pure_python(inp, ker)
    python_ms = (time.perf_counter() - t0) / N_py * 1000

    try:
        import torch
        import torch.nn.functional as F
        inp_pt = torch.tensor(inp, dtype=torch.float32)
        ker_pt = torch.tensor(ker, dtype=torch.float32)
        for _ in range(5):
            F.conv2d(inp_pt, ker_pt)
        t0 = time.perf_counter()
        for _ in range(100):
            F.conv2d(inp_pt, ker_pt)
        pytorch_ms = (time.perf_counter() - t0) / 100 * 1000
        pytorch_line = f"PyTorch (CPU):    {pytorch_ms:.3f}ms"
    except ImportError:
        pytorch_line = "PyTorch: not installed"

    print(f"Input:  {inp.shape}  Kernel: {ker.shape}\n")
    print(f"Pure Python:      {python_ms:.1f}ms")
    print(f"TitanGrad (Numba):{numba_ms:.3f}ms")
    print(pytorch_line)
    print(f"\nSpeedup vs Python: {python_ms/numba_ms:.0f}x")