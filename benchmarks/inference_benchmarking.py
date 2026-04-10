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
from typing import Tuple, Dict, Optional, List
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
from nltools.algorithms.backends import check_gpu_available

# Try to import visualization libraries
try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    HAS_VIS = True
except ImportError:
    HAS_VIS = False

# Try to import tqdm for progress bars
try:
    import tqdm  # noqa: F401

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
    parallel: Optional[str],
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark one_sample_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = one_sample_permutation_test(
        data,
        n_permute=n_permute,
        parallel=parallel,
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
    parallel: Optional[str],
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
        parallel=parallel,
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
    parallel: Optional[str],
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
        parallel=parallel,
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
    parallel: Optional[str],
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
        parallel=parallel,
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
    parallel: Optional[str],
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
        parallel=parallel,
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
    parallel: Optional[str],
    n_jobs: int = -1,
    random_state: Optional[int] = 42,
) -> Tuple[float, float]:
    """Benchmark isc_permutation_test."""
    mem_start = get_memory_mb()

    start = time.perf_counter()
    _ = isc_permutation_test(
        data,
        n_permute=n_permute,
        parallel=parallel,
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
    parallel: Optional[str],
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
        parallel=parallel,
        n_jobs=n_jobs,
        random_state=random_state,
    )
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


# ============================================================================
# Helper Functions for Benchmark Execution
# ============================================================================


def get_device_label(device: str, n_jobs: Optional[int] = None) -> str:
    """Get human-readable label for device matching parallel kwarg semantics.

    Parameters
    ----------
    device : str
        Device identifier ('numpy', 'cpu-parallel', 'torch')
    n_jobs : int, optional
        Number of CPU cores used (for cpu-parallel)

    Returns
    -------
    str
        Descriptive label matching parallel kwarg semantics
    """
    if device == "numpy":
        return "CPU Single (parallel=None)"
    elif device == "cpu-parallel":
        if n_jobs is None or n_jobs == -1:
            import os

            n_jobs = os.cpu_count() or 1
        return f"CPU Parallel (parallel='cpu', n_jobs={n_jobs})"
    elif device == "torch":
        return "GPU (parallel='gpu')"
    else:
        return device


def get_algorithm_label(algorithm: str, method: Optional[str] = None) -> str:
    """Get human-readable label for algorithm.

    Parameters
    ----------
    algorithm : str
        Algorithm name (e.g., 'one_sample', 'isc', 'timeseries_correlation')
    method : str, optional
        Method/variant name (e.g., 'circle_shift', 'bootstrap', 'permute')

    Returns
    -------
    str
        Descriptive label (e.g., 'One Sample Permutation', 'ISC Bootstrap')
    """
    algorithm_labels = {
        "one_sample": "One Sample Permutation",
        "two_sample": "Two Sample Permutation",
        "correlation": "Correlation Permutation",
        "timeseries_correlation": "Timeseries Correlation Permutation",
        "matrix": "Matrix Permutation (Mantel Test)",
        "isc": "ISC Permutation",
        "isc_group": "ISC Group Permutation",
    }

    base_label = algorithm_labels.get(algorithm, algorithm.replace("_", " ").title())

    # Add method/variant information if provided
    if method:
        method_labels = {
            "circle_shift": "Circle Shift",
            "phase_randomize": "Phase Randomize",
            "bootstrap": "Bootstrap",
            "permute": "Permutation",
        }
        method_label = method_labels.get(method, method.replace("_", " ").title())

        # For timeseries correlation, method is part of the name
        if algorithm == "timeseries_correlation":
            return f"{base_label} ({method_label})"
        # For ISC, add method info
        elif algorithm == "isc":
            return f"{base_label} ({method_label})"
        elif algorithm == "isc_group":
            return f"{base_label} ({method_label})"

    return base_label


def print_device_header(
    device: str,
    device_num: int,
    total_devices: int,
    quiet: bool = False,
    n_jobs: Optional[int] = None,
):
    """Print header for new device section."""
    if quiet:
        return
    print("\n" + "=" * 80)
    print(f"DEVICE {device_num}/{total_devices}: {get_device_label(device, n_jobs)}")
    print("=" * 80)


def print_test_header(
    algorithm: str,
    test_label: str,
    test_num: int,
    total_tests: int,
    quiet: bool = False,
    method: Optional[str] = None,
):
    """Print header for new test."""
    if quiet:
        return
    algorithm_label = get_algorithm_label(algorithm, method)
    print(f"\n[{test_num}/{total_tests}] {algorithm_label}: {test_label}")


def print_test_result(
    algorithm: str,
    test_label: str,
    device: str,
    time_seconds: float,
    memory_mb: float,
    quiet: bool = False,
    n_jobs: Optional[int] = None,
):
    """Print result for a single test."""
    if quiet:
        return
    device_label = get_device_label(device, n_jobs)
    print(
        f"  {device_label:50s} | Time: {time_seconds:10.4f}s | Memory: {memory_mb:9.4f} MB"
    )


def print_device_summary(
    device: str,
    device_results: List[Dict],
    quiet: bool = False,
    n_jobs: Optional[int] = None,
):
    """Print summary of results for current device."""
    if quiet or not device_results:
        return

    device_label = get_device_label(device, n_jobs)
    print(f"\n{device_label} Summary:")
    print("-" * 80)

    # Group by algorithm
    df_device = pd.DataFrame(device_results)
    for algorithm in df_device["algorithm"].unique():
        alg_df = df_device[df_device["algorithm"] == algorithm]
        avg_time = alg_df["time_seconds"].mean()
        avg_mem = alg_df["memory_mb"].mean()
        print(
            f"  {algorithm:25s} | Avg Time: {avg_time:8.4f}s | Avg Memory: {avg_mem:7.4f} MB"
        )
    print("-" * 80)


def print_overall_progress(current: int, total: int, quiet: bool = False):
    """Print overall progress indicator."""
    if quiet:
        return
    percentage = (current / total * 100) if total > 0 else 0
    remaining = total - current
    print(f"\n{'=' * 80}")
    print(
        f"OVERALL PROGRESS: {current}/{total} complete ({percentage:.1f}%) | {remaining} remaining"
    )
    print("=" * 80)


# ============================================================================
# Benchmark Execution Functions (by algorithm)
# ============================================================================


def run_one_sample_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    test_counter: Dict,
    quiet: bool = False,
) -> int:
    """Run one_sample benchmarks for given device."""
    tests_run = 0

    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        if config.get("quick", False) and n_features > 10000:
            continue

        tests_run += 1
        test_label = f"n={n_samples}, f={n_features}, p={n_permute}"
        test_counter["current"] += 1

        print_test_header(
            "one_sample",
            test_label,
            test_counter["current"],
            test_counter["total"],
            quiet,
        )

        # Generate data
        data = generate_one_sample_data(n_samples, n_features)

        condition_key = f"one_sample_{n_samples}_{n_features}_{n_permute}"

        # Run benchmark
        time_result, mem_result = benchmark_one_sample(
            data,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

        # Store result
        result_dict = {
            "algorithm": "one_sample",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

        # Calculate speedups if baselines exist
        if device == "cpu-parallel" and condition_key in numpy_baselines:
            result_dict["speedup_vs_numpy"] = (
                numpy_baselines[condition_key] / time_result
            )
            result_dict["speedup_vs_cpu_parallel"] = 1.0
        elif device == "torch":
            if condition_key in numpy_baselines:
                result_dict["speedup_vs_numpy"] = (
                    numpy_baselines[condition_key] / time_result
                )
            if condition_key in cpu_parallel_baselines:
                result_dict["speedup_vs_cpu_parallel"] = (
                    cpu_parallel_baselines[condition_key] / time_result
                )
        elif device == "numpy":
            result_dict["speedup_vs_numpy"] = 1.0
            numpy_baselines[condition_key] = time_result

        if device == "cpu-parallel":
            cpu_parallel_baselines[condition_key] = time_result

        results.append(result_dict)

        # Get n_jobs for printing
        import os

        print_n_jobs = os.cpu_count() or 1 if device == "cpu-parallel" else None

        print_test_result(
            "One Sample Permutation",
            test_label,
            device,
            time_result,
            mem_result,
            quiet,
            print_n_jobs,
        )
        del data

    return tests_run


def run_two_sample_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    test_counter: Dict,
    quiet: bool = False,
) -> int:
    """Run two_sample benchmarks for given device."""
    tests_run = 0

    # Two-sample: use same n_samples for both groups (from config["n_samples"])
    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        n_samples1 = n_samples
        n_samples2 = n_samples
        if config.get("quick", False) and n_features > 10000:
            continue

        tests_run += 1
        test_label = f"n={n_samples}, f={n_features}, p={n_permute}"
        test_counter["current"] += 1

        print_test_header(
            "two_sample",
            test_label,
            test_counter["current"],
            test_counter["total"],
            quiet,
        )

        # Generate data
        data1, data2 = generate_two_sample_data(n_samples1, n_samples2, n_features)

        condition_key = f"two_sample_{n_samples}_{n_features}_{n_permute}"

        # Run benchmark
        time_result, mem_result = benchmark_two_sample(
            data1,
            data2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

        # Store result
        result_dict = {
            "algorithm": "two_sample",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

        # Calculate speedups
        if device == "cpu-parallel" and condition_key in numpy_baselines:
            result_dict["speedup_vs_numpy"] = (
                numpy_baselines[condition_key] / time_result
            )
            result_dict["speedup_vs_cpu_parallel"] = 1.0
        elif device == "torch":
            if condition_key in numpy_baselines:
                result_dict["speedup_vs_numpy"] = (
                    numpy_baselines[condition_key] / time_result
                )
            if condition_key in cpu_parallel_baselines:
                result_dict["speedup_vs_cpu_parallel"] = (
                    cpu_parallel_baselines[condition_key] / time_result
                )
        elif device == "numpy":
            result_dict["speedup_vs_numpy"] = 1.0
            numpy_baselines[condition_key] = time_result

        if device == "cpu-parallel":
            cpu_parallel_baselines[condition_key] = time_result

        results.append(result_dict)

        # Get n_jobs for printing
        import os

        print_n_jobs = os.cpu_count() or 1 if device == "cpu-parallel" else None

        print_test_result(
            "Two Sample Permutation",
            test_label,
            device,
            time_result,
            mem_result,
            quiet,
            print_n_jobs,
        )
        del data1, data2

    return tests_run


def run_correlation_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    test_counter: Dict,
    quiet: bool = False,
) -> int:
    """Run correlation benchmarks for given device."""
    tests_run = 0

    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        if config.get("quick", False) and n_features > 10000:
            continue

        tests_run += 1
        test_label = f"n={n_samples}, f={n_features}, p={n_permute}"
        test_counter["current"] += 1

        print_test_header(
            "correlation",
            test_label,
            test_counter["current"],
            test_counter["total"],
            quiet,
        )

        # Generate data
        data1, data2 = generate_correlation_data(n_samples, n_features)

        condition_key = f"correlation_{n_samples}_{n_features}_{n_permute}"

        # Run benchmark
        time_result, mem_result = benchmark_correlation(
            data1,
            data2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

        # Store result
        result_dict = {
            "algorithm": "correlation",
            "n_samples": n_samples,
            "n_features": n_features,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

        # Calculate speedups
        if device == "cpu-parallel" and condition_key in numpy_baselines:
            result_dict["speedup_vs_numpy"] = (
                numpy_baselines[condition_key] / time_result
            )
            result_dict["speedup_vs_cpu_parallel"] = 1.0
        elif device == "torch":
            if condition_key in numpy_baselines:
                result_dict["speedup_vs_numpy"] = (
                    numpy_baselines[condition_key] / time_result
                )
            if condition_key in cpu_parallel_baselines:
                result_dict["speedup_vs_cpu_parallel"] = (
                    cpu_parallel_baselines[condition_key] / time_result
                )
        elif device == "numpy":
            result_dict["speedup_vs_numpy"] = 1.0
            numpy_baselines[condition_key] = time_result

        if device == "cpu-parallel":
            cpu_parallel_baselines[condition_key] = time_result

        results.append(result_dict)

        # Get n_jobs for printing
        import os

        print_n_jobs = os.cpu_count() or 1 if device == "cpu-parallel" else None

        print_test_result(
            "Correlation Permutation",
            test_label,
            device,
            time_result,
            mem_result,
            quiet,
            print_n_jobs,
        )
        del data1, data2

    return tests_run


def run_timeseries_correlation_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    test_counter: Dict,
    quiet: bool = False,
) -> int:
    """Run timeseries_correlation benchmarks for given device."""
    tests_run = 0
    methods = ["circle_shift", "phase_randomize"]

    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        if config.get("quick", False) and n_features > 10000:
            continue

        # Timeseries correlation uses 1D data (single feature)
        if n_features != 1:
            continue

        for method in methods:
            tests_run += 1
            test_label = f"{method}, n={n_samples}, p={n_permute}"
            test_counter["current"] += 1

            print_test_header(
                "timeseries_correlation",
                test_label,
                test_counter["current"],
                test_counter["total"],
                quiet,
                method=method,
            )

            # Generate 1D time series data
            rng = np.random.RandomState(42)
            t = np.linspace(0, 10 * np.pi, n_samples)
            data1 = np.sin(t) + rng.randn(n_samples) * 0.1
            data2 = np.cos(t) + rng.randn(n_samples) * 0.1
            data1 = data1.astype(np.float32)
            data2 = data2.astype(np.float32)

            condition_key = f"timeseries_correlation_{method}_{n_samples}_{n_permute}"

            # Run benchmark
            time_result, mem_result = benchmark_timeseries_correlation(
                data1,
                data2,
                n_permute,
                method,
                parallel=parallel,
                n_jobs=-1 if device == "cpu-parallel" else None,
                random_state=42,
            )

            # Store result
            result_dict = {
                "algorithm": f"timeseries_correlation_{method}",
                "n_samples": n_samples,
                "n_features": 1,
                "n_permute": n_permute,
                "backend": device,
                "time_seconds": time_result,
                "memory_mb": mem_result,
                "speedup_vs_numpy": None,
                "speedup_vs_cpu_parallel": None,
            }

            # Calculate speedups (no numpy baseline for timeseries)
            if device == "torch" and condition_key in cpu_parallel_baselines:
                result_dict["speedup_vs_cpu_parallel"] = (
                    cpu_parallel_baselines[condition_key] / time_result
                )
            elif device == "cpu-parallel":
                result_dict["speedup_vs_cpu_parallel"] = 1.0
                cpu_parallel_baselines[condition_key] = time_result

            results.append(result_dict)

            # Get n_jobs for printing
            import os

            print_n_jobs = os.cpu_count() or 1 if device == "cpu-parallel" else None

            print_test_result(
                f"Timeseries Correlation ({method.replace('_', ' ').title()})",
                test_label,
                device,
                time_result,
                mem_result,
                quiet,
                print_n_jobs,
            )
            del data1, data2

    return tests_run


def run_matrix_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    test_counter: Dict,
    quiet: bool = False,
) -> int:
    """Run matrix benchmarks for given device."""
    tests_run = 0

    # Matrix supports numpy and cpu-parallel
    if device not in ["numpy", "cpu-parallel"]:
        return 0

    # Matrix size is set by n_samples flag
    for matrix_size, n_permute in product(config["n_samples"], config["n_permute"]):
        if config.get("quick", False) and matrix_size > 100:
            continue

        tests_run += 1
        test_label = f"size={matrix_size}, p={n_permute}"
        test_counter["current"] += 1

        print_test_header(
            "matrix",
            test_label,
            test_counter["current"],
            test_counter["total"],
            quiet,
        )

        # Generate symmetric matrices
        matrix1, matrix2 = generate_matrix_data(matrix_size)

        # Run benchmark
        time_result, mem_result = benchmark_matrix(
            matrix1,
            matrix2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

        condition_key = f"matrix_{matrix_size}_{n_permute}"

        # Store result
        result_dict = {
            "algorithm": "matrix",
            "n_samples": matrix_size,
            "n_features": matrix_size,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

        # Calculate speedups
        if device == "cpu-parallel" and condition_key in numpy_baselines:
            result_dict["speedup_vs_numpy"] = (
                numpy_baselines[condition_key] / time_result
            )
            result_dict["speedup_vs_cpu_parallel"] = 1.0
        elif device == "numpy":
            result_dict["speedup_vs_numpy"] = 1.0
            numpy_baselines[condition_key] = time_result

        if device == "cpu-parallel":
            cpu_parallel_baselines[condition_key] = time_result

        results.append(result_dict)

        # Get n_jobs for printing
        import os

        print_n_jobs = os.cpu_count() or 1 if device == "cpu-parallel" else None

        print_test_result(
            "Matrix Permutation (Mantel Test)",
            test_label,
            device,
            time_result,
            mem_result,
            quiet,
            print_n_jobs,
        )
        del matrix1, matrix2

    return tests_run


def run_isc_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    test_counter: Dict,
    quiet: bool = False,
) -> int:
    """Run isc benchmarks for given device."""
    tests_run = 0
    n_subjects = [10, 20, 30]
    n_timepoints = config.get("n_timepoints", [100, 500])
    n_voxels_config = config["n_features"][:3]

    for n_subj, n_tp, n_vox, n_permute in product(
        n_subjects, n_timepoints, n_voxels_config, config["n_permute"]
    ):
        if config.get("quick", False) and n_vox > 1000:
            continue

        tests_run += 1
        test_label = (
            f"subjects={n_subj}, timepoints={n_tp}, voxels={n_vox}, p={n_permute}"
        )
        test_counter["current"] += 1

        print_test_header(
            "isc",
            test_label,
            test_counter["current"],
            test_counter["total"],
            quiet,
            method="bootstrap",  # Default method for ISC
        )

        # Generate ISC data: (n_timepoints, n_subjects, n_voxels)
        data = generate_isc_data(n_tp, n_subj, n_vox)

        condition_key = f"isc_{n_subj}_{n_tp}_{n_vox}_{n_permute}"

        # Run benchmark
        time_result, mem_result = benchmark_isc(
            data,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

        # Store result
        result_dict = {
            "algorithm": "isc",
            "n_samples": n_tp,
            "n_features": n_vox,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

        # Calculate speedups (no numpy baseline for isc)
        if device == "torch" and condition_key in cpu_parallel_baselines:
            result_dict["speedup_vs_cpu_parallel"] = (
                cpu_parallel_baselines[condition_key] / time_result
            )
        elif device == "cpu-parallel":
            result_dict["speedup_vs_cpu_parallel"] = 1.0
            cpu_parallel_baselines[condition_key] = time_result

        results.append(result_dict)

        # Get n_jobs for printing
        import os

        print_n_jobs = os.cpu_count() or 1 if device == "cpu-parallel" else None

        print_test_result(
            "ISC Permutation (Bootstrap)",
            test_label,
            device,
            time_result,
            mem_result,
            quiet,
            print_n_jobs,
        )
        del data

    return tests_run


def run_isc_group_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    test_counter: Dict,
    quiet: bool = False,
) -> int:
    """Run isc_group benchmarks for given device."""
    tests_run = 0
    group_sizes = [(10, 10), (20, 20), (30, 30)]
    n_timepoints = config.get("n_timepoints", [100, 500])
    n_voxels_config = config["n_features"][:3]

    # ISC group supports numpy and cpu-parallel
    if device not in ["numpy", "cpu-parallel"]:
        return 0

    for (n_subj1, n_subj2), n_tp, n_vox, n_permute in product(
        group_sizes, n_timepoints, n_voxels_config, config["n_permute"]
    ):
        if config.get("quick", False) and n_vox > 1000:
            continue

        tests_run += 1
        test_label = f"group1={n_subj1}, group2={n_subj2}, timepoints={n_tp}, voxels={n_vox}, p={n_permute}"
        test_counter["current"] += 1

        print_test_header(
            "isc_group",
            test_label,
            test_counter["current"],
            test_counter["total"],
            quiet,
            method="permute",  # Default method for ISC group
        )

        # Generate ISC data for two groups
        group1 = generate_isc_data(n_tp, n_subj1, n_vox)
        group2 = generate_isc_data(n_tp, n_subj2, n_vox)

        condition_key = f"isc_group_{n_subj1}_{n_subj2}_{n_tp}_{n_vox}_{n_permute}"

        # Run benchmark
        time_result, mem_result = benchmark_isc_group(
            group1,
            group2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

        # Store result
        result_dict = {
            "algorithm": "isc_group",
            "n_samples": f"{n_tp},{n_subj1},{n_subj2}",
            "n_features": n_vox,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

        # Calculate speedups
        if device == "cpu-parallel" and condition_key in numpy_baselines:
            result_dict["speedup_vs_numpy"] = (
                numpy_baselines[condition_key] / time_result
            )
            result_dict["speedup_vs_cpu_parallel"] = 1.0
        elif device == "numpy":
            result_dict["speedup_vs_numpy"] = 1.0
            numpy_baselines[condition_key] = time_result

        if device == "cpu-parallel":
            cpu_parallel_baselines[condition_key] = time_result

        results.append(result_dict)

        # Get n_jobs for printing
        import os

        print_n_jobs = os.cpu_count() or 1 if device == "cpu-parallel" else None

        print_test_result(
            "ISC Group Permutation",
            test_label,
            device,
            time_result,
            mem_result,
            quiet,
            print_n_jobs,
        )
        del group1, group2

    return tests_run


# ============================================================================
# Main Benchmark Runner
# ============================================================================


def count_total_tests(
    config: Dict, gpu_available: bool, no_gpu: bool
) -> Dict[str, int]:
    """Count total number of tests to run, grouped by device."""
    device_counts = {"numpy": 0, "cpu-parallel": 0, "torch": 0}

    for algorithm in config["algorithms"]:
        if algorithm == "one_sample":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "two_sample":
            # Two-sample: use same n_samples for both groups
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "correlation":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "timeseries_correlation":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if config.get("quick", False) and n_features > 10000:
                    continue
                if n_features == 1:
                    device_counts["cpu-parallel"] += 2  # Two methods
                    if gpu_available and not no_gpu:
                        device_counts["torch"] += 2
        elif algorithm == "matrix":
            # Matrix size is set by n_samples flag
            for matrix_size, n_permute in product(
                config["n_samples"], config["n_permute"]
            ):
                if config.get("quick", False) and matrix_size > 100:
                    continue
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
        elif algorithm == "isc":
            n_subjects = [10, 20, 30]
            n_timepoints = config.get("n_timepoints", [100, 500])
            n_voxels = config["n_features"][:3]
            for n_subj, n_tp, n_vox, n_permute in product(
                n_subjects, n_timepoints, n_voxels, config["n_permute"]
            ):
                if config.get("quick", False) and n_vox > 1000:
                    continue
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "isc_group":
            group_sizes = [(10, 10), (20, 20), (30, 30)]
            n_timepoints = config.get("n_timepoints", [100, 500])
            n_voxels = config["n_features"][:3]
            for (n1, n2), n_tp, n_vox, n_permute in product(
                group_sizes, n_timepoints, n_voxels, config["n_permute"]
            ):
                if config.get("quick", False) and n_vox > 1000:
                    continue
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1

    total = sum(device_counts.values())
    return {"device_counts": device_counts, "total": total}


def run_systematic_benchmarks(config: Dict) -> pd.DataFrame:
    """
    Run systematic benchmark grid for inference algorithms, grouped by device.

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

    # Count total tests
    test_counts = count_total_tests(config, gpu_available, config.get("no_gpu", False))
    total_tests = test_counts["total"]
    device_counts = test_counts["device_counts"]

    if not config.get("quiet", False):
        print(f"\nTotal benchmarks to run: {total_tests}")
        print(f"  - CPU (NumPy): {device_counts['numpy']}")
        print(f"  - CPU-Parallel: {device_counts['cpu-parallel']}")
        if gpu_available and not config.get("no_gpu", False):
            print(f"  - GPU (PyTorch): {device_counts['torch']}")
        print()

    # Results storage
    results = []

    # Track baselines for speedup calculation
    numpy_baselines = {}
    cpu_parallel_baselines = {}

    # Test counter for progress tracking
    test_counter = {"current": 0, "total": total_tests}

    # Determine which devices to run
    devices_to_run = []
    if device_counts["numpy"] > 0:
        devices_to_run.append(("numpy", None))
    if device_counts["cpu-parallel"] > 0:
        devices_to_run.append(("cpu-parallel", "cpu"))
    if device_counts["torch"] > 0:
        devices_to_run.append(("torch", "gpu"))

    # Iterate through devices first, then algorithms
    for device_idx, (device, parallel) in enumerate(devices_to_run, 1):
        # Get n_jobs for cpu-parallel device
        if device == "cpu-parallel":
            import os

            n_jobs = os.cpu_count() or 1
        else:
            n_jobs = None

        print_device_header(
            device, device_idx, len(devices_to_run), config.get("quiet", False), n_jobs
        )

        device_results = []

        # Run all algorithms for this device
        for algorithm in config["algorithms"]:
            if algorithm == "one_sample":
                tests_run = run_one_sample_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    test_counter,
                    config.get("quiet", False),
                )
                device_results.extend(results[-tests_run:])

            elif algorithm == "two_sample":
                tests_run = run_two_sample_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    test_counter,
                    config.get("quiet", False),
                )
                device_results.extend(results[-tests_run:])

            elif algorithm == "correlation":
                tests_run = run_correlation_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    test_counter,
                    config.get("quiet", False),
                )
                device_results.extend(results[-tests_run:])

            elif algorithm == "timeseries_correlation":
                tests_run = run_timeseries_correlation_benchmarks(
                    config,
                    device,
                    parallel,
                    cpu_parallel_baselines,
                    results,
                    test_counter,
                    config.get("quiet", False),
                )
                device_results.extend(results[-tests_run:])

            elif algorithm == "matrix":
                tests_run = run_matrix_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    test_counter,
                    config.get("quiet", False),
                )
                device_results.extend(results[-tests_run:])

            elif algorithm == "isc":
                tests_run = run_isc_benchmarks(
                    config,
                    device,
                    parallel,
                    cpu_parallel_baselines,
                    results,
                    test_counter,
                    config.get("quiet", False),
                )
                device_results.extend(results[-tests_run:])

            elif algorithm == "isc_group":
                tests_run = run_isc_group_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    test_counter,
                    config.get("quiet", False),
                )
                device_results.extend(results[-tests_run:])

        # Print device summary
        print_device_summary(device, device_results, config.get("quiet", False), n_jobs)

        # Print overall progress
        print_overall_progress(
            test_counter["current"], test_counter["total"], config.get("quiet", False)
        )

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

        # Average speedups - filter out None/NaN values
        cpu_parallel_df = alg_df[alg_df["backend"] == "cpu-parallel"]
        gpu_df = alg_df[alg_df["backend"] == "torch"]

        if len(cpu_parallel_df) > 0:
            # Filter out None/NaN values before calculating mean
            speedup_cp_numpy = cpu_parallel_df["speedup_vs_numpy"].dropna()
            if len(speedup_cp_numpy) > 0:
                avg_speedup_cp = speedup_cp_numpy.mean()
                print(f"  CPU-parallel avg speedup vs NumPy: {avg_speedup_cp:.4f}x")
            else:
                print("  CPU-parallel avg speedup vs NumPy: N/A (no numpy baseline)")

        if len(gpu_df) > 0:
            # Filter out None/NaN values before calculating mean
            speedup_gpu_numpy = gpu_df["speedup_vs_numpy"].dropna()
            speedup_gpu_cp = gpu_df["speedup_vs_cpu_parallel"].dropna()

            if len(speedup_gpu_numpy) > 0:
                avg_speedup_gpu_np = speedup_gpu_numpy.mean()
                print(f"  GPU avg speedup vs NumPy: {avg_speedup_gpu_np:.4f}x")
            else:
                print("  GPU avg speedup vs NumPy: N/A (no numpy baseline)")

            if len(speedup_gpu_cp) > 0:
                avg_speedup_gpu_cp = speedup_gpu_cp.mean()
                print(f"  GPU avg speedup vs CPU-parallel: {avg_speedup_gpu_cp:.4f}x")
            else:
                print(
                    "  GPU avg speedup vs CPU-parallel: N/A (no cpu-parallel baseline)"
                )


def create_benchmark_plot(df: pd.DataFrame, output_path: str):
    """Create a seaborn stripplot visualization of benchmark results.

    Similar to seaborn's jitter stripplot example, showing individual
    observations with conditional means.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results DataFrame
    output_path : str
        Path to save the figure
    """
    if not HAS_VIS:
        print("Warning: seaborn/matplotlib not available. Skipping visualization.")
        return

    # Prepare data for plotting
    plot_df = df.copy()

    # Create a combined problem size label
    plot_df["problem_size"] = (
        plot_df["algorithm"].astype(str)
        + "\n"
        + "n="
        + plot_df["n_samples"].astype(str)
        + ", "
        + "f="
        + plot_df["n_features"].astype(str)
        + ", "
        + "p="
        + plot_df["n_permute"].astype(str)
    )

    # Map backend names for clearer labels
    backend_map = {
        "numpy": "CPU (NumPy)",
        "cpu-parallel": "CPU-Parallel",
        "torch": "GPU (PyTorch)",
    }
    plot_df["backend_label"] = plot_df["backend"].map(backend_map)

    # Initialize figure
    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots(figsize=(14, 8))
    sns.despine(bottom=True, left=True)

    # Show individual observations with stripplot
    sns.stripplot(
        data=plot_df,
        x="time_seconds",
        y="problem_size",
        hue="backend_label",
        dodge=True,
        alpha=0.25,
        zorder=1,
        legend=False,
        size=3,
    )

    # Show conditional means with pointplot
    # Adjust width based on number of hue levels
    n_hue_levels = plot_df["backend_label"].nunique()
    dodge_width = 0.8 - 0.8 / n_hue_levels

    sns.pointplot(
        data=plot_df,
        x="time_seconds",
        y="problem_size",
        hue="backend_label",
        dodge=dodge_width,
        palette="dark",
        errorbar=None,
        markers="d",
        markersize=6,
        linestyle="none",
        ax=ax,
    )

    # Improve legend
    sns.move_legend(
        ax,
        loc="lower right",
        ncol=3,
        frameon=True,
        columnspacing=1,
        handletextpad=0,
        title="Backend",
    )

    # Labels and formatting
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Algorithm & Problem Size", fontsize=12)
    ax.set_title(
        "Inference Algorithm Benchmark Results", fontsize=14, fontweight="bold"
    )

    # Use log scale for x-axis if range is wide
    if plot_df["time_seconds"].max() / plot_df["time_seconds"].min() > 100:
        ax.set_xscale("log")
        ax.set_xlabel("Time (seconds, log scale)", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to: {output_path}")


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
        default="one_sample,two_sample,correlation,timeseries_correlation,matrix,isc,isc_group",
        help="Algorithms to benchmark (comma-separated): one_sample, two_sample, "
        "correlation, timeseries_correlation, matrix, isc, isc_group "
        "(default: one_sample,two_sample,correlation,timeseries_correlation,matrix,isc,isc_group)",
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=str,
        default="25",
        help="Sample sizes (comma-separated, default: 25)",
    )

    parser.add_argument(
        "-f",
        "--n-features",
        type=str,
        default="100",
        help="Feature/voxel counts (comma-separated, default: 100)",
    )

    parser.add_argument(
        "-t",
        "--n-timepoints",
        type=str,
        default="100,500",
        help="Timepoints for ISC tests (comma-separated, default: 100,500)",
    )

    parser.add_argument(
        "-p",
        "--n-permute",
        type=str,
        default="5000",
        help="Permutation counts (comma-separated, default: 5000)",
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

    try:
        n_timepoints = [int(x.strip()) for x in args.n_timepoints.split(",")]
    except ValueError:
        print(
            f"Error: Invalid n-timepoints argument '{args.n_timepoints}'. Must be comma-separated integers."
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
        "n_timepoints": n_timepoints,
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

        # Count tests
        test_counts = count_total_tests(config, gpu_available, args.no_gpu)
        print(f"\nTotal benchmarks: {test_counts['total']}")
        print(f"  - CPU (NumPy): {test_counts['device_counts']['numpy']}")
        print(f"  - CPU-Parallel: {test_counts['device_counts']['cpu-parallel']}")
        if gpu_available and not args.no_gpu:
            print(f"  - GPU (PyTorch): {test_counts['device_counts']['torch']}")

        print("\nTo run: Remove --dry-run flag")
        print("=" * 80)
        return

    # Run benchmarks
    results_df = run_systematic_benchmarks(config)

    # Save to CSV with timestamp
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate unique filenames
    csv_base = args.output.replace(".csv", "")
    csv_filename = f"{csv_base}_{timestamp}.csv"
    csv_path = os.path.join(os.path.dirname(__file__), csv_filename)
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to: {csv_path}")

    # Generate visualization
    if HAS_VIS:
        fig_base = csv_filename.replace(".csv", "")
        fig_filename = f"{fig_base}.png"
        fig_path = os.path.join(os.path.dirname(__file__), fig_filename)
        create_benchmark_plot(results_df, fig_path)
    else:
        print("Warning: seaborn/matplotlib not available. Skipping visualization.")

    # Print summary
    if not args.quiet:
        print_summary(results_df)

        # Show full results table with 4 decimal precision
        print(f"\n{'=' * 80}")
        print("FULL RESULTS TABLE")
        print(f"{'=' * 80}\n")

        # Format numeric columns to 4 decimal places
        display_df = results_df.copy()
        numeric_cols = [
            "time_seconds",
            "memory_mb",
            "speedup_vs_numpy",
            "speedup_vs_cpu_parallel",
        ]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(
                    lambda x: f"{x:.4f}" if pd.notna(x) else "N/A"
                )

        print(display_df.to_string(index=False))


if __name__ == "__main__":
    main()
