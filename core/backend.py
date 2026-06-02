def get_backend(device='cpu'):
    if device == 'cpu':
        import numpy as np
        return np
    elif device == 'gpu' or device == 'cuda':
        try:
            import cupy as cp
            return cp
        except ImportError:
            raise RuntimeError(
                "CuPy not installed. Install with: pip install cupy-cuda12x"
                "\n(Use the version matching your CUDA installation)"
            )
    else:
        raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'gpu'")
