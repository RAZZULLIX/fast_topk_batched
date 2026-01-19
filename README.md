# Fast TopK

High-performance batched Top-K selection for CPU inference. Optimized for LLM sampling workloads.

## Performance

**Up to 4x faster than PyTorch CPU, competitive with CUDA for small batches.**

### Benchmarks (K=50)

...

| Implementation | Batch=1, Vocab=128K | Batch=64, Vocab=128K |
|----------------|---------------------|----------------------|
| Fast TopK      | 0.057 ms           | 2.10 ms              |
| PyTorch CPU    | 0.777 ms           | 7.16 ms              |
| PyTorch CUDA   | 0.086 ms           | 0.375 ms             |

**llama.cpp integration:** 63% faster prompt processing (pp512: 81→142 t/s on RTX 3090)

## Installation

**Pre-built binaries:** See `bin/` directory

**Build from source:**
Windows
```bash
gcc -shared -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -finline-functions -fomit-frame-pointer -static -static-libgcc fast_topk_batched.c -o fast_topk_batched.dll -lwinmm
```


```bash
gcc -shared -fPIC -O3 -march=native -mtune=native -flto -ffast-math -funroll-loops -finline-functions -fomit-frame-pointer fast_topk_batched.c -o libfast_topk.so
```

## Usage

```python
import ctypes
import numpy as np

lib = ctypes.CDLL('./libfast_topk.so')
lib.fast_topk_batched.argtypes = [
    ctypes.POINTER(ctypes.c_float),
    ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.POINTER(ctypes.c_int)
]

# batch_size=16, vocab_size=128000, k=50
logits = np.random.randn(16, 128000).astype(np.float32)
indices = np.zeros(16 * 50, dtype=np.int32)

lib.fast_topk_batched(
    logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
    16, 128000, 50,
    indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
)

indices = indices.reshape(16, 50)  # Top-50 indices per sequence
```

## How It Works

- Adaptive sampling + min-heap tracking
- AVX2 SIMD for 8-wide parallel comparisons
- Cache-optimized block scanning
- Fast paths for sorted/constant inputs

## Files

- `fast_topk_batched.c` - Main implementation
- `bin/` - Pre-compiled libraries
- `examples/llama-cpp/` - llama.cpp integration
- `benchmarks/topk_l10_audit.py` - Validation suite

## License

MIT
