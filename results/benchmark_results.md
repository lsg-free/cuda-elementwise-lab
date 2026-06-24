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

## Reduction Benchmark

Data size: 16,777,216 elements  
Data type: FP32  
Threads per block: 256  
Repeat: 100  

| Op | Avg Time (ms) | Bandwidth (GB/s) | Result | Expected | Check |
|---|---:|---:|---:|---:|---|
| Sum | 0.153658 | 438.452 | 1.67772e+07 | 1.67772e+07 | PASSED |
| Max | 0.152695 | 441.217 | 499 | 499 | PASSED |

Note: Bandwidth is estimated from global memory traffic across all reduction levels.