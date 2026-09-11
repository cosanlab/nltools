---
title: Benchmarks
description: GPU vs CPU speed and memory tables for the ridge, predict, and inference domains.
---

# Benchmarks

The tables below are generated from a committed `benchmarks/results/*.parquet` artifact
(one per host, with an `.env.json` recording Python, NumPy, PyTorch, and nltools
versions) and are not re-run on each doc build, so treat absolute timings as
indicative rather than current. Regenerate them with `uv run python -m
benchmarks.run` followed by `uv run python -m benchmarks.build_docs`.

<!-- BENCH:START -->
!!! note "Auto-generated"
    This block is generated from `benchmarks/results/*.parquet` by `uv run python -m benchmarks.build_docs`. Do not edit by hand. One section per host — an MPS run and a CUDA run coexist.

### `Eshin-M3-Air`

**Host:** `Eshin-M3-Air`  
**Platform:** macOS-15.7.4-arm64-arm-64bit  
**Python:** 3.11.14  
**NumPy:** 2.4.4  
**PyTorch:** 2.11.0  
**nltools:** 0.5.1  
**GPU:** MPS

**GPU speedup**

#### ridge — GPU speedup (mps)

| Condition | CPU | MPS | Speedup |
|---|--:|--:|--:|
| `ridge_cv[1000x20000f100]` | 15.15 s | 14.71 s | **1.03×** |
| `ridge_cv[500x20000f50]` | 12.96 s | 12.89 s | **1.01×** |

#### inference — GPU speedup (mps)

| Condition | CPU | MPS | Speedup |
|---|--:|--:|--:|
| `correlation[perm=1000]` | 76.7 ms | 68.3 ms | **1.12×** |
| `correlation[perm=3000]` | 110.6 ms | 193.6 ms | **0.57×** |
| `one_sample[perm=1000]` | 310.9 ms | 179.3 ms | **1.73×** |
| `one_sample[perm=3000]` | 576.3 ms | 438.5 ms | **1.31×** |
| `two_sample[perm=1000]` | 264.9 ms | 173.3 ms | **1.53×** |
| `two_sample[perm=3000]` | 397.9 ms | 501.5 ms | **0.79×** |

**Full results**

#### ridge

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| `BrainData.fit[ridge,200x20000]` | cpu | 239.0 ms | 0.1 MB | - |
| `ridge_cv[1000x20000f100]` | cpu | 15.15 s | 0.0 MB | - |
| `ridge_cv[1000x20000f100]` | mps | 14.71 s | 0.2 MB | - |
| `ridge_cv[500x20000f50]` | cpu | 12.96 s | 0.0 MB | - |
| `ridge_cv[500x20000f50]` | mps | 12.89 s | 0.1 MB | - |

#### predict

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| `roi[50parcels]` | cpu | 1.27 s | 0.5 MB | - |
| `searchlight[60x400]` | cpu | 452.0 ms | 0.2 MB | - |
| `whole_brain[200x20000]` | cpu | 630.2 ms | 0.0 MB | - |

#### inference

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| `correlation[perm=1000]` | cpu | 76.7 ms | 0.0 MB | - |
| `correlation[perm=1000]` | mps | 68.3 ms | 0.0 MB | - |
| `correlation[perm=3000]` | cpu | 110.6 ms | 0.0 MB | - |
| `correlation[perm=3000]` | mps | 193.6 ms | 0.2 MB | - |
| `one_sample[perm=1000]` | cpu | 310.9 ms | 0.6 MB | - |
| `one_sample[perm=1000]` | mps | 179.3 ms | 0.0 MB | - |
| `one_sample[perm=3000]` | cpu | 576.3 ms | 16.6 MB | - |
| `one_sample[perm=3000]` | mps | 438.5 ms | 0.0 MB | - |
| `two_sample[perm=1000]` | cpu | 264.9 ms | 1.5 MB | - |
| `two_sample[perm=1000]` | mps | 173.3 ms | 1.9 MB | - |
| `two_sample[perm=3000]` | cpu | 397.9 ms | 10.9 MB | - |
| `two_sample[perm=3000]` | mps | 501.5 ms | 2.9 MB | - |

### `pikachu.ucsd.edu`

**Host:** `pikachu.ucsd.edu`  
**Platform:** Linux-6.17.0-1031-nvidia-aarch64-with-glibc2.39  
**Python:** 3.11.15  
**NumPy:** 2.4.6  
**PyTorch:** 2.13.0+cu130  
**nltools:** 0.5.1 @ 55e44f06  
**GPU:** CUDA

**GPU speedup**

#### ridge — GPU speedup (cuda)

| Condition | CPU | CUDA | Speedup |
|---|--:|--:|--:|
| `ridge_cv[1000x20000f100]` | 19.25 s | 13.32 s | **1.45×** |
| `ridge_cv[500x20000f50]` | 16.34 s | 12.05 s | **1.36×** |

#### inference — GPU speedup (cuda)

| Condition | CPU | CUDA | Speedup |
|---|--:|--:|--:|
| `correlation[perm=1000]` | 122.4 ms | 60.3 ms | **2.03×** |
| `correlation[perm=3000]` | 184.8 ms | 179.9 ms | **1.03×** |
| `one_sample[perm=1000]` | 433.5 ms | 98.7 ms | **4.39×** |
| `one_sample[perm=3000]` | 774.0 ms | 279.9 ms | **2.77×** |
| `two_sample[perm=1000]` | 285.9 ms | 97.5 ms | **2.93×** |
| `two_sample[perm=3000]` | 609.9 ms | 285.3 ms | **2.14×** |

**Full results**

#### ridge

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| `BrainData.fit[ridge,200x20000]` | cpu | 683.5 ms | 63.9 MB | - |
| `ridge_cv[1000x20000f100]` | cpu | 19.25 s | 223.9 MB | - |
| `ridge_cv[1000x20000f100]` | cuda | 13.32 s | 144.0 MB | 138 MB |
| `ridge_cv[500x20000f50]` | cpu | 16.34 s | 175.9 MB | - |
| `ridge_cv[500x20000f50]` | cuda | 12.05 s | 40.0 MB | 86 MB |

#### predict

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| `roi[50parcels]` | cpu | 882.9 ms | 0.0 MB | - |
| `searchlight[60x400]` | cpu | 253.1 ms | 0.3 MB | - |
| `whole_brain[200x20000]` | cpu | 589.2 ms | 64.0 MB | - |

#### inference

| Condition | Device | Time | Peak RSS | GPU mem |
|---|---|--:|--:|--:|
| `correlation[perm=1000]` | cpu | 122.4 ms | 0.0 MB | - |
| `correlation[perm=1000]` | cuda | 60.3 ms | 0.4 MB | 34 MB |
| `correlation[perm=3000]` | cpu | 184.8 ms | 0.0 MB | - |
| `correlation[perm=3000]` | cuda | 179.9 ms | 0.0 MB | 35 MB |
| `one_sample[perm=1000]` | cpu | 433.5 ms | 40.4 MB | - |
| `one_sample[perm=1000]` | cuda | 98.7 ms | 21.0 MB | 655 MB |
| `one_sample[perm=3000]` | cpu | 774.0 ms | 241.4 MB | - |
| `one_sample[perm=3000]` | cuda | 279.9 ms | 180.8 MB | 1895 MB |
| `two_sample[perm=1000]` | cpu | 285.9 ms | 61.1 MB | - |
| `two_sample[perm=1000]` | cuda | 97.5 ms | 21.0 MB | 77 MB |
| `two_sample[perm=3000]` | cpu | 609.9 ms | 240.1 MB | - |
| `two_sample[perm=3000]` | cuda | 285.3 ms | 180.8 MB | 158 MB |
<!-- BENCH:END -->
