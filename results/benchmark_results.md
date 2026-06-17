# Benchmark Results

## Environment

- GPU: NVIDIA GeForce RTX 5070
- Data size: 16,777,216 elements
- Repeat: 100
- Default block size: 256 threads/block

## Block Size Sweep

| Block Size | Avg Time (ms) | Bandwidth (GB/s) |
|---|---:|---:|
| 128 | 0.390745 | 515.238 |
| 256 | 0.391640 | 514.060 |
| 512 | 0.392257 | 513.252 |
| 1024 | 0.399230 | 504.287 |

## Float4 Vectorization

| Kernel | Avg Time (ms) | Bandwidth (GB/s) | Check |
|---|---:|---:|---|
| naive | 0.393375 | 511.793 | PASSED |
| float4 | 0.393677 | 511.400 | PASSED |

Speedup: 0.999x

## Half2 Vectorization

| Kernel | Avg Time (ms) | Bandwidth (GB/s) | Check |
|---|---:|---:|---|
| half | 0.208287 | 483.290 | PASSED |
| half2 | 0.179846 | 559.720 | PASSED |

Speedup: 1.158x