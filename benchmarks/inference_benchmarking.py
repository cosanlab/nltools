"""
Systematic benchmarking of inference algorithms for neuroimaging workflows.

Compares CPU (NumPy), CPU-parallel (joblib), and GPU (PyTorch) implementations
across realistic problem sizes and permutation counts.

Usage:
    # Dry run with defaults
    python inference_benchmarking.py --dry-run

    # Single algorithm, small problems
    python inference_benchmarking.py --algorithm one_sample --n-features "1,100"

    # Custom configuration
    python inference_benchmarking.py --algorithm one_sample,two_sample --n-permute 5000

    # Quick test (skip large problems)
    python inference_benchmarking.py --quick
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
import platform
import argparse
from typing import Tuple, Dict, Optional, Union
from itertools import product

# Import nltools components
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nltools.algorithms.inference import (
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    timeseries_correlation_permutation_test,
    matrix_permutation_test,
    isc_permutation_test,
    isc_group_permutation_test,
)
from nltools.backends import Backend, check_gpu_available

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


# ============================================================================
# Data Generation Functions
# ============================================================================


def generate_one_sample_data(
    n_samples: int, n_features: int, random_state: Optional[int] = 42
) -> np.ndarray:
    """Generate synthetic data for one-sample test."""
    rng = np.random.RandomState(random_state)
    return rng.randn(n_samples, n_features).astype(np.float32)


def generate_two_sample_data(
    n_samples1: int,
    n_samples2: int,
    n_features: int,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for two-sample test."""
    rng = np.random.RandomState(random_state)
    return (
        rng.randn(n_samples1, n_features).astype(np.float32),
        rng.randn(n_samples2, n_features).astype(np.float32),
    )


def generate_correlation_data(
    n_samples: int, n_features: int, random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic correlated data for correlation test."""
    rng = np.random.RandomState(random_state)
    # Create correlated data
    base = rng.randn(n_samples, n_features)
    noise = rng.randn(n_samples, n_features) * 0.3
    data1 = base
    data2 = base + noise
    return data1.astype(np.float32), data2.astype(np.float32)


def generate_matrix_data(
    matrix_size: int, random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic symmetric matrices for matrix permutation test."""
    rng = np.random.RandomState(random_state)
    # Create symmetric matrices
    base1 = rng.randn(matrix_size, matrix_size)
    base2 = rng.randn(matrix_size, matrix_size)
    matrix1 = (base1 + base1.T) / 2
    matrix2 = (base2 + base2.T) / 2
    return matrix1.astype(np.float32), matrix2.astype(np.float32)


def generate_isc_data(
    n_timepoints: int,
    n_subjects: int,
    n_voxels: int,
    random_state: Optional[int] = 42,
) -> np.ndarray:
    """Generate synthetic timeseries data for ISC test."""
    rng = np.random.RandomState(random_state)
    # Shape: (n_timepoints, n_subjects, n_voxels)
    return rng.randn(n_timepoints, n_subjects, n_voxels).astype(np.float32)


# ============================================================================
# Benchmark Functions
# ============================================================================


def benchmark_one_sample(
    data: np.ndarray,
    n_permute: int,
    backend: Union[Backend, str, None],
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark one_sample_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = one_sample_permutation_test(
        data,
        n_permute=n_permute,
        backend=backend,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def benchmark_two_sample(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    backend: Union[Backend, str, None],
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark two_sample_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = two_sample_permutation_test(
        data1,
        data2,
        n_permute=n_permute,
        backend=backend,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def benchmark_correlation(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    backend: Union[Backend, str, None],
    metric: str = "pearson",
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark correlation_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = correlation_permutation_test(
        data1,
        data2,
        n_permute=n_permute,
        metric=metric,
        backend=backend,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def benchmark_timeseries_correlation(
    data1: np.ndarray,
    data2: np.ndarray,
    n_permute: int,
    method: str,
    backend: Union[Backend, str, None],
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark timeseries_correlation_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = timeseries_correlation_permutation_test(
        data1,
        data2,
        method=method,
        n_permute=n_permute,
        backend=backend,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def benchmark_matrix(
    matrix1: np.ndarray,
    matrix2: np.ndarray,
    n_permute: int,
    metric: str = "pearson",
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark matrix_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = matrix_permutation_test(
        matrix1,
        matrix2,
        n_permute=n_permute,
        metric=metric,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def benchmark_isc(
    data: np.ndarray,
    n_permute: int,
    backend: Union[Backend, str, None],
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark isc_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = isc_permutation_test(
        data,
        n_permute=n_permute,
        backend=backend,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def benchmark_isc_group(
    group1: np.ndarray,
    group2: np.ndarray,
    n_permute: int,
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark isc_group_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = isc_group_permutation_test(
        group1,
        group2,
        n_permute=n_permute,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


# ============================================================================
# CLI and Configuration
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic inference algorithm benchmarks for neuroimaging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with defaults
  %(prog)s --dry-run

  # Single algorithm, small problems
  %(prog)s --algorithm one_sample --n-features "1,100"

  # Custom permutation count
  %(prog)s --n-permute 5000

  # Quick test (skip large problems)
  %(prog)s --quick

  # Skip GPU
  %(prog)s --no-gpu
        """,
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default="one_sample,two_sample,correlation",
        help="Algorithms to benchmark (comma-separated): one_sample, two_sample, "
        "correlation, timeseries_correlation, matrix, isc, isc_group "
        "(default: one_sample,two_sample,correlation)",
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=str,
        default="50,200,500",
        help="Sample sizes (comma-separated, default: 50,200,500)",
    )

    parser.add_argument(
        "-f",
        "--n-features",
        type=str,
        default="1,100,1000",
        help="Feature/voxel counts (comma-separated, default: 1,100,1000)",
    )

    parser.add_argument(
        "-p",
        "--n-permute",
        type=str,
        default="1000,5000",
        help="Permutation counts (comma-separated, default: 1000,5000)",
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: skip large problems (n_features > 10000)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show benchmark plan without running",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU benchmarks (CPU and CPU-parallel only)",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results_inference_systematic.csv",
        help="Output CSV filename (default: results_inference_systematic.csv)",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output, no progress bars",
    )

    return parser.parse_args()


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def run_systematic_benchmarks(config: Dict) -> pd.DataFrame:
    """
    Run systematic benchmark grid for inference algorithms.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys:
        - algorithms: list of algorithm names
        - n_samples: list of sample sizes
        - n_features: list of feature counts
        - n_permute: list of permutation counts
        - quick: bool (skip large problems)
        - no_gpu: bool
        - quiet: bool

    Returns
    -------
    results : pd.DataFrame
        Benchmark results with columns: algorithm, n_samples, n_features,
        n_permute, backend, time_seconds, memory_mb, speedup_vs_numpy,
        speedup_vs_cpu_parallel
    """
    if not config.get("quiet", False):
        print("=" * 80)
        print("Systematic Inference Algorithm Benchmarks for Neuroimaging")
        print("=" * 80)

    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()

    if not config.get("quiet", False):
        print(f"\nGPU Available: {gpu_available}")
        if gpu_available:
            print(f"Device: {gpu_info['device']}")
            print(f"Device Name: {gpu_info['device_name']}")

        # Print system info
        print(f"\nSystem: {platform.system()} {platform.release()}")
        print(f"Machine: {platform.machine()}")
        print(f"Python: {platform.python_version()}")

        import torch

        print(f"NumPy: {np.__version__}")
        print(f"PyTorch: {torch.__version__}")

        print("\n" + "=" * 80)
        print("Benchmark Grid:")
        print(f"- Algorithms: {', '.join(config['algorithms'])}")
        print(f"- Samples: {', '.join(str(x) for x in config['n_samples'])}")
        print(f"- Features: {', '.join(str(x) for x in config['n_features'])}")
        print(f"- Permutations: {', '.join(str(x) for x in config['n_permute'])}")
        print("=" * 80 + "\n")

    # Results storage
    results = []

    # Track baselines for speedup calculation
    numpy_baselines = {}
    cpu_parallel_baselines = {}

    # Build condition list
    total_conditions = 0
    for algorithm in config["algorithms"]:
        if algorithm == "one_sample":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                total_conditions += 1
        elif algorithm == "two_sample":
            # Two-sample uses group sizes
            group_sizes = [(20, 25), (50, 50), (100, 100)]
            for (n1, n2), n_features, n_permute in product(
                group_sizes, config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                total_conditions += 1
        elif algorithm == "correlation":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                total_conditions += 1
        elif algorithm == "timeseries_correlation":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                total_conditions += 1
                total_conditions += 1  # Two methods: circle_shift, phase_randomize
        elif algorithm == "matrix":
            matrix_sizes = [20, 50, 100]
            for matrix_size, n_permute in product(matrix_sizes, config["n_permute"]):
                total_conditions += 1
        elif algorithm == "isc":
            n_subjects = [10, 20, 30]
            n_timepoints = [100, 500]
            n_voxels = config["n_features"][:3]  # Limit for ISC
            for n_subj, n_tp, n_vox, n_permute in product(
                n_subjects, n_timepoints, n_voxels, config["n_permute"]
            ):
                if config.get("quick", False) and n_vox > 10000:
                    continue
                total_conditions += 1
        elif algorithm == "isc_group":
            group_sizes = [(10, 10), (20, 20), (30, 30)]
            n_timepoints = [100, 500]
            n_voxels = config["n_features"][:3]
            for (n1, n2), n_tp, n_vox, n_permute in product(
                group_sizes, n_timepoints, n_voxels, config["n_permute"]
            ):
                if config.get("quick", False) and n_vox > 10000:
                    continue
                total_conditions += 1

    # Set up progress bar
    use_progress = HAS_TQDM and not config.get("quiet", False)
    if use_progress:
        pbar = tqdm(total=total_conditions, desc="Benchmarking", unit="cond")
    else:
        pbar = None

    condition_num = 0

    # Iterate through algorithms
    for algorithm in config["algorithms"]:
        if algorithm == "one_sample":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue

                condition_num += 1
                if pbar:
                    pbar.set_description(
                        f"one_sample: n={n_samples}, f={n_features}, p={n_permute}"
                    )

                # Generate data
                data = generate_one_sample_data(n_samples, n_features)

                # Benchmark backends
                condition_key = f"one_sample_{n_samples}_{n_features}_{n_permute}"

                # CPU (NumPy)
                time_np, mem_np = benchmark_one_sample(
                    data, n_permute, backend="numpy", random_state=42
                )
                results.append(
                    {
                        "algorithm": "one_sample",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_permute": n_permute,
                        "backend": "numpy",
                        "time_seconds": time_np,
                        "memory_mb": mem_np,
                        "speedup_vs_numpy": 1.0,
                        "speedup_vs_cpu_parallel": None,
                    }
                )
                numpy_baselines[condition_key] = time_np

                # CPU-parallel
                time_cp, mem_cp = benchmark_one_sample(
                    data, n_permute, backend=None, n_jobs=-1, random_state=42
                )
                speedup_cp = numpy_baselines[condition_key] / time_cp
                results.append(
                    {
                        "algorithm": "one_sample",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_permute": n_permute,
                        "backend": "cpu-parallel",
                        "time_seconds": time_cp,
                        "memory_mb": mem_cp,
                        "speedup_vs_numpy": speedup_cp,
                        "speedup_vs_cpu_parallel": 1.0,
                    }
                )
                cpu_parallel_baselines[condition_key] = time_cp

                # GPU (if available)
                skip_gpu = config.get("no_gpu", False) or not gpu_available
                if not skip_gpu:
                    time_gpu, mem_gpu = benchmark_one_sample(
                        data, n_permute, backend="torch", random_state=42
                    )
                    speedup_gpu_np = numpy_baselines[condition_key] / time_gpu
                    speedup_gpu_cp = cpu_parallel_baselines[condition_key] / time_gpu
                    results.append(
                        {
                            "algorithm": "one_sample",
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "n_permute": n_permute,
                            "backend": "torch",
                            "time_seconds": time_gpu,
                            "memory_mb": mem_gpu,
                            "speedup_vs_numpy": speedup_gpu_np,
                            "speedup_vs_cpu_parallel": speedup_gpu_cp,
                        }
                    )

                if pbar:
                    pbar.update(1)
                del data

        elif algorithm == "two_sample":
            group_sizes = [(20, 25), (50, 50), (100, 100)]
            for (n_samples1, n_samples2), n_features, n_permute in product(
                group_sizes, config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue

                condition_num += 1
                if pbar:
                    pbar.set_description(
                        f"two_sample: n1={n_samples1}, n2={n_samples2}, f={n_features}, p={n_permute}"
                    )

                # Generate data
                data1, data2 = generate_two_sample_data(
                    n_samples1, n_samples2, n_features
                )

                condition_key = (
                    f"two_sample_{n_samples1}_{n_samples2}_{n_features}_{n_permute}"
                )

                # CPU (NumPy)
                time_np, mem_np = benchmark_two_sample(
                    data1, data2, n_permute, backend="numpy", random_state=42
                )
                results.append(
                    {
                        "algorithm": "two_sample",
                        "n_samples": f"{n_samples1},{n_samples2}",
                        "n_features": n_features,
                        "n_permute": n_permute,
                        "backend": "numpy",
                        "time_seconds": time_np,
                        "memory_mb": mem_np,
                        "speedup_vs_numpy": 1.0,
                        "speedup_vs_cpu_parallel": None,
                    }
                )
                numpy_baselines[condition_key] = time_np

                # CPU-parallel
                time_cp, mem_cp = benchmark_two_sample(
                    data1, data2, n_permute, backend=None, n_jobs=-1, random_state=42
                )
                speedup_cp = numpy_baselines[condition_key] / time_cp
                results.append(
                    {
                        "algorithm": "two_sample",
                        "n_samples": f"{n_samples1},{n_samples2}",
                        "n_features": n_features,
                        "n_permute": n_permute,
                        "backend": "cpu-parallel",
                        "time_seconds": time_cp,
                        "memory_mb": mem_cp,
                        "speedup_vs_numpy": speedup_cp,
                        "speedup_vs_cpu_parallel": 1.0,
                    }
                )
                cpu_parallel_baselines[condition_key] = time_cp

                # GPU (if available)
                skip_gpu = config.get("no_gpu", False) or not gpu_available
                if not skip_gpu:
                    time_gpu, mem_gpu = benchmark_two_sample(
                        data1, data2, n_permute, backend="torch", random_state=42
                    )
                    speedup_gpu_np = numpy_baselines[condition_key] / time_gpu
                    speedup_gpu_cp = cpu_parallel_baselines[condition_key] / time_gpu
                    results.append(
                        {
                            "algorithm": "two_sample",
                            "n_samples": f"{n_samples1},{n_samples2}",
                            "n_features": n_features,
                            "n_permute": n_permute,
                            "backend": "torch",
                            "time_seconds": time_gpu,
                            "memory_mb": mem_gpu,
                            "speedup_vs_numpy": speedup_gpu_np,
                            "speedup_vs_cpu_parallel": speedup_gpu_cp,
                        }
                    )

                if pbar:
                    pbar.update(1)
                del data1, data2

        elif algorithm == "correlation":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue

                condition_num += 1
                if pbar:
                    pbar.set_description(
                        f"correlation: n={n_samples}, f={n_features}, p={n_permute}"
                    )

                # Generate data
                data1, data2 = generate_correlation_data(n_samples, n_features)

                condition_key = f"correlation_{n_samples}_{n_features}_{n_permute}"

                # CPU (NumPy)
                time_np, mem_np = benchmark_correlation(
                    data1, data2, n_permute, backend="numpy", random_state=42
                )
                results.append(
                    {
                        "algorithm": "correlation",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_permute": n_permute,
                        "backend": "numpy",
                        "time_seconds": time_np,
                        "memory_mb": mem_np,
                        "speedup_vs_numpy": 1.0,
                        "speedup_vs_cpu_parallel": None,
                    }
                )
                numpy_baselines[condition_key] = time_np

                # CPU-parallel
                time_cp, mem_cp = benchmark_correlation(
                    data1, data2, n_permute, backend=None, n_jobs=-1, random_state=42
                )
                speedup_cp = numpy_baselines[condition_key] / time_cp
                results.append(
                    {
                        "algorithm": "correlation",
                        "n_samples": n_samples,
                        "n_features": n_features,
                        "n_permute": n_permute,
                        "backend": "cpu-parallel",
                        "time_seconds": time_cp,
                        "memory_mb": mem_cp,
                        "speedup_vs_numpy": speedup_cp,
                        "speedup_vs_cpu_parallel": 1.0,
                    }
                )
                cpu_parallel_baselines[condition_key] = time_cp

                # GPU (if available)
                skip_gpu = config.get("no_gpu", False) or not gpu_available
                if not skip_gpu:
                    time_gpu, mem_gpu = benchmark_correlation(
                        data1, data2, n_permute, backend="torch", random_state=42
                    )
                    speedup_gpu_np = numpy_baselines[condition_key] / time_gpu
                    speedup_gpu_cp = cpu_parallel_baselines[condition_key] / time_gpu
                    results.append(
                        {
                            "algorithm": "correlation",
                            "n_samples": n_samples,
                            "n_features": n_features,
                            "n_permute": n_permute,
                            "backend": "torch",
                            "time_seconds": time_gpu,
                            "memory_mb": mem_gpu,
                            "speedup_vs_numpy": speedup_gpu_np,
                            "speedup_vs_cpu_parallel": speedup_gpu_cp,
                        }
                    )

                if pbar:
                    pbar.update(1)
                del data1, data2

        elif algorithm == "timeseries_correlation":
            methods = ["circle_shift", "phase_randomize"]
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue

                # Timeseries correlation uses 1D data (single feature)
                # We'll test with n_features=1 for each method
                if n_features != 1:
                    continue

                for method in methods:
                    condition_num += 1
                    if pbar:
                        pbar.set_description(
                            f"timeseries_correlation ({method}): n={n_samples}, p={n_permute}"
                        )

                    # Generate 1D time series data
                    rng = np.random.RandomState(42)
                    # Create correlated time series
                    t = np.linspace(0, 10 * np.pi, n_samples)
                    data1 = np.sin(t) + rng.randn(n_samples) * 0.1
                    data2 = np.cos(t) + rng.randn(n_samples) * 0.1
                    data1 = data1.astype(np.float32)
                    data2 = data2.astype(np.float32)

                    condition_key = (
                        f"timeseries_correlation_{method}_{n_samples}_{n_permute}"
                    )

                    # CPU-parallel (timeseries_correlation doesn't support NumPy backend)
                    time_cp, mem_cp = benchmark_timeseries_correlation(
                        data1,
                        data2,
                        n_permute,
                        method,
                        backend=None,
                        n_jobs=-1,
                        random_state=42,
                    )
                    results.append(
                        {
                            "algorithm": f"timeseries_correlation_{method}",
                            "n_samples": n_samples,
                            "n_features": 1,  # Always 1D for timeseries
                            "n_permute": n_permute,
                            "backend": "cpu-parallel",
                            "time_seconds": time_cp,
                            "memory_mb": mem_cp,
                            "speedup_vs_numpy": None,  # No NumPy baseline
                            "speedup_vs_cpu_parallel": 1.0,
                        }
                    )
                    cpu_parallel_baselines[condition_key] = time_cp

                    # GPU (if available)
                    skip_gpu = config.get("no_gpu", False) or not gpu_available
                    if not skip_gpu:
                        time_gpu, mem_gpu = benchmark_timeseries_correlation(
                            data1,
                            data2,
                            n_permute,
                            method,
                            backend="torch",
                            random_state=42,
                        )
                        speedup_gpu_cp = (
                            cpu_parallel_baselines[condition_key] / time_gpu
                        )
                        results.append(
                            {
                                "algorithm": f"timeseries_correlation_{method}",
                                "n_samples": n_samples,
                                "n_features": 1,
                                "n_permute": n_permute,
                                "backend": "torch",
                                "time_seconds": time_gpu,
                                "memory_mb": mem_gpu,
                                "speedup_vs_numpy": None,
                                "speedup_vs_cpu_parallel": speedup_gpu_cp,
                            }
                        )

                    if pbar:
                        pbar.update(1)
                    del data1, data2

        elif algorithm == "matrix":
            matrix_sizes = [20, 50, 100]
            for matrix_size, n_permute in product(matrix_sizes, config["n_permute"]):
                if config.get("quick", False) and matrix_size > 50:
                    continue

                condition_num += 1
                if pbar:
                    pbar.set_description(f"matrix: size={matrix_size}, p={n_permute}")

                # Generate symmetric matrices
                matrix1, matrix2 = generate_matrix_data(matrix_size)

                condition_key = f"matrix_{matrix_size}_{n_permute}"

                # Matrix permutation only supports CPU-parallel
                time_cp, mem_cp = benchmark_matrix(
                    matrix1, matrix2, n_permute, n_jobs=-1, random_state=42
                )
                results.append(
                    {
                        "algorithm": "matrix",
                        "n_samples": matrix_size,  # Matrix size
                        "n_features": matrix_size,  # Same as size
                        "n_permute": n_permute,
                        "backend": "cpu-parallel",
                        "time_seconds": time_cp,
                        "memory_mb": mem_cp,
                        "speedup_vs_numpy": None,  # No NumPy baseline
                        "speedup_vs_cpu_parallel": 1.0,
                    }
                )

                if pbar:
                    pbar.update(1)
                del matrix1, matrix2

        elif algorithm == "isc":
            n_subjects = [10, 20, 30]
            n_timepoints = [100, 500]
            # Limit voxels for ISC (uses first 3 from config)
            n_voxels_config = config["n_features"][:3]
            for n_subj, n_tp, n_vox, n_permute in product(
                n_subjects, n_timepoints, n_voxels_config, config["n_permute"]
            ):
                if config.get("quick", False) and n_vox > 1000:
                    continue

                condition_num += 1
                if pbar:
                    pbar.set_description(
                        f"isc: subjects={n_subj}, timepoints={n_tp}, voxels={n_vox}, p={n_permute}"
                    )

                # Generate ISC data: (n_timepoints, n_subjects, n_voxels)
                data = generate_isc_data(n_tp, n_subj, n_vox)

                condition_key = f"isc_{n_subj}_{n_tp}_{n_vox}_{n_permute}"

                # CPU-parallel (ISC supports CPU-parallel and GPU)
                time_cp, mem_cp = benchmark_isc(
                    data, n_permute, backend=None, n_jobs=-1, random_state=42
                )
                results.append(
                    {
                        "algorithm": "isc",
                        "n_samples": n_tp,  # Timepoints
                        "n_features": n_vox,  # Voxels
                        "n_permute": n_permute,
                        "backend": "cpu-parallel",
                        "time_seconds": time_cp,
                        "memory_mb": mem_cp,
                        "speedup_vs_numpy": None,  # No NumPy baseline
                        "speedup_vs_cpu_parallel": 1.0,
                    }
                )
                cpu_parallel_baselines[condition_key] = time_cp

                # GPU (if available)
                skip_gpu = config.get("no_gpu", False) or not gpu_available
                if not skip_gpu:
                    time_gpu, mem_gpu = benchmark_isc(
                        data, n_permute, backend="torch", random_state=42
                    )
                    speedup_gpu_cp = cpu_parallel_baselines[condition_key] / time_gpu
                    results.append(
                        {
                            "algorithm": "isc",
                            "n_samples": n_tp,
                            "n_features": n_vox,
                            "n_permute": n_permute,
                            "backend": "torch",
                            "time_seconds": time_gpu,
                            "memory_mb": mem_gpu,
                            "speedup_vs_numpy": None,
                            "speedup_vs_cpu_parallel": speedup_gpu_cp,
                        }
                    )

                if pbar:
                    pbar.update(1)
                del data

        elif algorithm == "isc_group":
            group_sizes = [(10, 10), (20, 20), (30, 30)]
            n_timepoints = [100, 500]
            n_voxels_config = config["n_features"][:3]
            for (n_subj1, n_subj2), n_tp, n_vox, n_permute in product(
                group_sizes, n_timepoints, n_voxels_config, config["n_permute"]
            ):
                if config.get("quick", False) and n_vox > 1000:
                    continue

                condition_num += 1
                if pbar:
                    pbar.set_description(
                        f"isc_group: group1={n_subj1}, group2={n_subj2}, timepoints={n_tp}, voxels={n_vox}, p={n_permute}"
                    )

                # Generate ISC data for two groups
                group1 = generate_isc_data(n_tp, n_subj1, n_vox)
                group2 = generate_isc_data(n_tp, n_subj2, n_vox)

                condition_key = (
                    f"isc_group_{n_subj1}_{n_subj2}_{n_tp}_{n_vox}_{n_permute}"
                )

                # ISC group only supports CPU-parallel
                time_cp, mem_cp = benchmark_isc_group(
                    group1, group2, n_permute, n_jobs=-1, random_state=42
                )
                results.append(
                    {
                        "algorithm": "isc_group",
                        "n_samples": f"{n_tp},{n_subj1},{n_subj2}",  # Timepoints, group sizes
                        "n_features": n_vox,
                        "n_permute": n_permute,
                        "backend": "cpu-parallel",
                        "time_seconds": time_cp,
                        "memory_mb": mem_cp,
                        "speedup_vs_numpy": None,
                        "speedup_vs_cpu_parallel": 1.0,
                    }
                )

                if pbar:
                    pbar.update(1)
                del group1, group2

    # Close progress bar
    if pbar:
        pbar.close()

    if not config.get("quiet", False):
        print(f"\n{'=' * 80}")
        print("Benchmark Complete!")
        print(f"{'=' * 80}")

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print formatted summary of benchmark results."""
    print(f"\n{'=' * 80}")
    print("SUMMARY TABLES")
    print(f"{'=' * 80}\n")

    # Group by algorithm
    for algorithm in df["algorithm"].unique():
        alg_df = df[df["algorithm"] == algorithm]
        print(f"\n{algorithm.upper()}:")
        print("-" * 80)

        # Average speedups
        cpu_parallel_df = alg_df[alg_df["backend"] == "cpu-parallel"]
        gpu_df = alg_df[alg_df["backend"] == "torch"]

        if len(cpu_parallel_df) > 0:
            avg_speedup_cp = cpu_parallel_df["speedup_vs_numpy"].mean()
            print(f"  CPU-parallel avg speedup vs NumPy: {avg_speedup_cp:.2f}x")

        if len(gpu_df) > 0:
            avg_speedup_gpu_np = gpu_df["speedup_vs_numpy"].mean()
            avg_speedup_gpu_cp = gpu_df["speedup_vs_cpu_parallel"].mean()
            print(f"  GPU avg speedup vs NumPy: {avg_speedup_gpu_np:.2f}x")
            print(f"  GPU avg speedup vs CPU-parallel: {avg_speedup_gpu_cp:.2f}x")


def main():
    """Run systematic benchmarks with CLI support and save results."""
    args = parse_args()

    # Parse comma-separated arguments
    try:
        algorithms = [x.strip() for x in args.algorithm.split(",")]
        valid_algorithms = [
            "one_sample",
            "two_sample",
            "correlation",
            "timeseries_correlation",
            "matrix",
            "isc",
            "isc_group",
        ]
        for alg in algorithms:
            if alg not in valid_algorithms:
                print(
                    f"Error: Invalid algorithm '{alg}'. Must be one of: {', '.join(valid_algorithms)}"
                )
                return
    except Exception as e:
        print(f"Error parsing algorithm argument: {e}")
        return

    try:
        n_samples = [int(x.strip()) for x in args.n_samples.split(",")]
    except ValueError:
        print(
            f"Error: Invalid n-samples argument '{args.n_samples}'. Must be comma-separated integers."
        )
        return

    try:
        n_features = [int(x.strip()) for x in args.n_features.split(",")]
    except ValueError:
        print(
            f"Error: Invalid n-features argument '{args.n_features}'. Must be comma-separated integers."
        )
        return

    try:
        n_permute = [int(x.strip()) for x in args.n_permute.split(",")]
    except ValueError:
        print(
            f"Error: Invalid n-permute argument '{args.n_permute}'. Must be comma-separated integers."
        )
        return

    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()

    # Build configuration
    config = {
        "algorithms": algorithms,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_permute": n_permute,
        "quick": args.quick,
        "no_gpu": args.no_gpu,
        "quiet": args.quiet,
        "output": args.output,
        "gpu_available": gpu_available,
        "gpu_device": gpu_info["device"] if gpu_available else "none",
    }

    # Handle dry-run
    if args.dry_run:
        print("=" * 80)
        print("DRY RUN: Inference Algorithm Benchmarks")
        print("=" * 80)
        print("\nConfiguration:")
        print(f"  Algorithms: {', '.join(algorithms)}")
        print(f"  Samples: {', '.join(str(x) for x in n_samples)}")
        print(f"  Features: {', '.join(str(x) for x in n_features)}")
        print(f"  Permutations: {', '.join(str(x) for x in n_permute)}")
        print(f"  Quick mode: {args.quick}")
        print(f"  GPU available: {gpu_available}")
        print(f"  Skip GPU: {args.no_gpu}")
        print("\nTo run: Remove --dry-run flag")
        print("=" * 80)
        return

    # Run benchmarks
    results_df = run_systematic_benchmarks(config)

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), args.output)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    if not args.quiet:
        print_summary(results_df)

        # Show full results table
        print(f"\n{'=' * 80}")
        print("FULL RESULTS TABLE")
        print(f"{'=' * 80}\n")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
