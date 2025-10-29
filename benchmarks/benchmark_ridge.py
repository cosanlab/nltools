"""
Comprehensive benchmarking suite for ridge regression performance.

Tests CPU (numpy) vs GPU (torch) backends across various problem sizes,
cross-validation scenarios, and real-world neuroimaging workflows.

Outputs: results_ridge_performance.csv
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
from typing import Dict, List, Tuple

# Import nltools components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nltools.algorithms.ridge import ridge_svd, ridge_cv
from nltools.backends import Backend, check_gpu_available, auto_select_backend


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_ridge_single(
    X: np.ndarray,
    y: np.ndarray,
    backend: Backend,
    alpha: float = 1.0
) -> Tuple[float, float]:
    """
    Benchmark a single ridge regression run.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    backend : Backend
        Backend to use for computation
    alpha : float
        Regularization parameter

    Returns
    -------
    time_seconds : float
        Execution time in seconds
    memory_mb : float
        Peak memory usage in MB
    """
    # Measure initial memory
    mem_start = get_memory_mb()

    # Time the computation
    start = time.perf_counter()
    coef = ridge_svd(X, y, alpha=alpha, backend=backend)
    end = time.perf_counter()

    # Measure peak memory
    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    time_seconds = end - start
    return time_seconds, memory_mb


def benchmark_ridge_cv_run(
    X: np.ndarray,
    y: np.ndarray,
    backend: Backend,
    cv: int = 5
) -> Tuple[float, float]:
    """
    Benchmark ridge regression with cross-validation.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    backend : Backend
        Backend to use
    cv : int
        Number of CV folds

    Returns
    -------
    time_seconds : float
        Execution time in seconds
    memory_mb : float
        Peak memory usage in MB
    """
    mem_start = get_memory_mb()

    start = time.perf_counter()
    result = ridge_cv(X, y, alphas=np.logspace(-2, 2, 10), cv=cv, backend=backend)
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    time_seconds = end - start
    return time_seconds, memory_mb


def run_benchmarks() -> pd.DataFrame:
    """
    Run comprehensive benchmark suite.

    Returns
    -------
    results : pd.DataFrame
        Benchmark results with columns:
        - scenario: Description of test scenario
        - backend: Backend used (numpy, torch-cuda, torch-mps, etc.)
        - n_samples: Number of samples
        - n_features: Number of features
        - cv_folds: Number of CV folds (1 = no CV)
        - time_seconds: Execution time
        - memory_mb: Memory usage
        - speedup_vs_numpy: Speedup compared to numpy baseline
    """
    print("=" * 80)
    print("Ridge Regression Performance Benchmark Suite")
    print("=" * 80)

    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()
    print(f"\nGPU Available: {gpu_available}")
    print(f"Device: {gpu_info['device']}")
    print(f"Device Name: {gpu_info['device_name']}\n")

    # Initialize results list
    results = []

    # Track numpy baseline times for speedup calculation
    numpy_baselines = {}

    # =========================================================================
    # Scenario 1: Basic Comparison - CPU vs GPU on medium dataset
    # =========================================================================
    print("-" * 80)
    print("Scenario 1: Basic CPU vs GPU Comparison (300×50k)")
    print("-" * 80)

    n_samples, n_features = 300, 50000
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)

    # Test NumPy backend
    print("  Testing numpy backend...")
    backend_numpy = Backend('numpy')
    time_np, mem_np = benchmark_ridge_single(X, y, backend_numpy)
    print(f"    Time: {time_np:.4f}s, Memory: {mem_np:.1f}MB")

    results.append({
        'scenario': 'basic_medium',
        'backend': 'numpy',
        'n_samples': n_samples,
        'n_features': n_features,
        'cv_folds': 1,
        'time_seconds': time_np,
        'memory_mb': mem_np,
        'speedup_vs_numpy': 1.0
    })
    numpy_baselines['basic_medium'] = time_np

    # Test Torch backend (if available)
    if gpu_available:
        print("  Testing torch backend...")
        backend_torch = Backend('torch')
        time_torch, mem_torch = benchmark_ridge_single(X, y, backend_torch)
        speedup = time_np / time_torch
        print(f"    Time: {time_torch:.4f}s, Memory: {mem_torch:.1f}MB, Speedup: {speedup:.1f}x")

        results.append({
            'scenario': 'basic_medium',
            'backend': backend_torch.name,
            'n_samples': n_samples,
            'n_features': n_features,
            'cv_folds': 1,
            'time_seconds': time_torch,
            'memory_mb': mem_torch,
            'speedup_vs_numpy': speedup
        })
    else:
        print("  Skipping torch backend (GPU not available)")

    # =========================================================================
    # Scenario 2: Problem Size Scaling - Small, Medium, Large
    # =========================================================================
    print("\n" + "-" * 80)
    print("Scenario 2: Problem Size Scaling")
    print("-" * 80)

    problem_sizes = [
        ('small', 100, 1000),
        ('medium', 300, 50000),
        ('large', 1000, 200000)
    ]

    for size_name, n_samp, n_feat in problem_sizes:
        print(f"\n  Testing {size_name}: {n_samp}×{n_feat}")
        X = np.random.randn(n_samp, n_feat).astype(np.float32)
        y = np.random.randn(n_samp).astype(np.float32)

        # NumPy
        print(f"    numpy...", end=" ")
        backend_np = Backend('numpy')
        time_np, mem_np = benchmark_ridge_single(X, y, backend_np)
        print(f"{time_np:.4f}s")

        scenario_key = f'scaling_{size_name}'
        results.append({
            'scenario': scenario_key,
            'backend': 'numpy',
            'n_samples': n_samp,
            'n_features': n_feat,
            'cv_folds': 1,
            'time_seconds': time_np,
            'memory_mb': mem_np,
            'speedup_vs_numpy': 1.0
        })
        numpy_baselines[scenario_key] = time_np

        # Torch (if available)
        if gpu_available:
            print(f"    torch...", end=" ")
            backend_torch = Backend('torch')
            time_torch, mem_torch = benchmark_ridge_single(X, y, backend_torch)
            speedup = time_np / time_torch
            print(f"{time_torch:.4f}s (speedup: {speedup:.1f}x)")

            results.append({
                'scenario': scenario_key,
                'backend': backend_torch.name,
                'n_samples': n_samp,
                'n_features': n_feat,
                'cv_folds': 1,
                'time_seconds': time_torch,
                'memory_mb': mem_torch,
                'speedup_vs_numpy': speedup
            })

    # =========================================================================
    # Scenario 3: Cross-Validation Impact - 1-fold vs 5-fold
    # =========================================================================
    print("\n" + "-" * 80)
    print("Scenario 3: Cross-Validation Overhead")
    print("-" * 80)

    n_samples, n_features = 300, 100000
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)

    for cv_folds in [1, 5]:
        print(f"\n  Testing {cv_folds}-fold CV")

        # NumPy
        print(f"    numpy...", end=" ")
        backend_np = Backend('numpy')
        if cv_folds == 1:
            time_np, mem_np = benchmark_ridge_single(X, y, backend_np)
        else:
            time_np, mem_np = benchmark_ridge_cv_run(X, y, backend_np, cv=cv_folds)
        print(f"{time_np:.4f}s")

        scenario_key = f'cv_{cv_folds}fold'
        results.append({
            'scenario': scenario_key,
            'backend': 'numpy',
            'n_samples': n_samples,
            'n_features': n_features,
            'cv_folds': cv_folds,
            'time_seconds': time_np,
            'memory_mb': mem_np,
            'speedup_vs_numpy': 1.0
        })
        numpy_baselines[scenario_key] = time_np

        # Torch (if available)
        if gpu_available:
            print(f"    torch...", end=" ")
            backend_torch = Backend('torch')
            if cv_folds == 1:
                time_torch, mem_torch = benchmark_ridge_single(X, y, backend_torch)
            else:
                time_torch, mem_torch = benchmark_ridge_cv_run(X, y, backend_torch, cv=cv_folds)
            speedup = time_np / time_torch
            print(f"{time_torch:.4f}s (speedup: {speedup:.1f}x)")

            results.append({
                'scenario': scenario_key,
                'backend': backend_torch.name,
                'n_samples': n_samples,
                'n_features': n_features,
                'cv_folds': cv_folds,
                'time_seconds': time_torch,
                'memory_mb': mem_torch,
                'speedup_vs_numpy': speedup
            })

    # =========================================================================
    # Scenario 4: Auto-Selection Validation
    # =========================================================================
    print("\n" + "-" * 80)
    print("Scenario 4: Auto Backend Selection Validation")
    print("-" * 80)

    auto_test_cases = [
        ('small_no_cv', 100, 1000, 1, 'numpy'),
        ('medium_with_cv', 300, 100000, 5, 'torch' if gpu_available else 'numpy'),
        ('large', 1000, 200000, 1, 'torch' if gpu_available else 'numpy')
    ]

    for case_name, n_samp, n_feat, cv_folds, expected_backend in auto_test_cases:
        print(f"\n  Testing {case_name}: {n_samp}×{n_feat}, CV={cv_folds}")
        X = np.random.randn(n_samp, n_feat).astype(np.float32)
        y = np.random.randn(n_samp).astype(np.float32)

        # Use auto_select_backend
        backend_auto = auto_select_backend(n_samp, n_feat, cv=cv_folds)
        print(f"    Expected: {expected_backend}, Selected: {backend_auto.name}")

        # Benchmark with auto-selected backend
        if cv_folds == 1:
            time_auto, mem_auto = benchmark_ridge_single(X, y, backend_auto)
        else:
            time_auto, mem_auto = benchmark_ridge_cv_run(X, y, backend_auto, cv=cv_folds)
        print(f"    Time: {time_auto:.4f}s, Memory: {mem_auto:.1f}MB")

        # Calculate speedup vs numpy baseline
        # We need to get numpy baseline for this case
        backend_np = Backend('numpy')
        if cv_folds == 1:
            time_np_baseline, _ = benchmark_ridge_single(X, y, backend_np)
        else:
            time_np_baseline, _ = benchmark_ridge_cv_run(X, y, backend_np, cv=cv_folds)

        speedup = time_np_baseline / time_auto

        results.append({
            'scenario': f'auto_{case_name}',
            'backend': backend_auto.name,
            'n_samples': n_samp,
            'n_features': n_feat,
            'cv_folds': cv_folds,
            'time_seconds': time_auto,
            'memory_mb': mem_auto,
            'speedup_vs_numpy': speedup
        })

    # =========================================================================
    # Scenario 5: Real-World Neuroimaging - Typical fMRI Analysis
    # =========================================================================
    print("\n" + "-" * 80)
    print("Scenario 5: Real-World Neuroimaging Workflows")
    print("-" * 80)

    real_world_cases = [
        ('whole_brain_prediction', 300, 100000, 5),
        ('searchlight_prep', 1000, 200000, 1)
    ]

    for case_name, n_samp, n_feat, cv_folds in real_world_cases:
        print(f"\n  Testing {case_name}: {n_samp}×{n_feat}, CV={cv_folds}")
        X = np.random.randn(n_samp, n_feat).astype(np.float32)
        y = np.random.randn(n_samp).astype(np.float32)

        # NumPy
        print(f"    numpy...", end=" ")
        backend_np = Backend('numpy')
        if cv_folds == 1:
            time_np, mem_np = benchmark_ridge_single(X, y, backend_np)
        else:
            time_np, mem_np = benchmark_ridge_cv_run(X, y, backend_np, cv=cv_folds)
        print(f"{time_np:.4f}s")

        scenario_key = f'realworld_{case_name}'
        results.append({
            'scenario': scenario_key,
            'backend': 'numpy',
            'n_samples': n_samp,
            'n_features': n_feat,
            'cv_folds': cv_folds,
            'time_seconds': time_np,
            'memory_mb': mem_np,
            'speedup_vs_numpy': 1.0
        })
        numpy_baselines[scenario_key] = time_np

        # Torch (if available)
        if gpu_available:
            print(f"    torch...", end=" ")
            backend_torch = Backend('torch')
            if cv_folds == 1:
                time_torch, mem_torch = benchmark_ridge_single(X, y, backend_torch)
            else:
                time_torch, mem_torch = benchmark_ridge_cv_run(X, y, backend_torch, cv=cv_folds)
            speedup = time_np / time_torch
            print(f"{time_torch:.4f}s (speedup: {speedup:.1f}x)")

            results.append({
                'scenario': scenario_key,
                'backend': backend_torch.name,
                'n_samples': n_samp,
                'n_features': n_feat,
                'cv_folds': cv_folds,
                'time_seconds': time_torch,
                'memory_mb': mem_torch,
                'speedup_vs_numpy': speedup
            })

    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)

    return pd.DataFrame(results)


def main():
    """Run benchmarks and save results to CSV."""
    # Run benchmark suite
    results_df = run_benchmarks()

    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), 'results_ridge_performance.csv')
    results_df.to_csv(output_path, index=False)

    print(f"\nResults saved to: {output_path}")
    print(f"\nSummary Statistics:")
    print(f"  Total tests run: {len(results_df)}")
    print(f"  Unique scenarios: {results_df['scenario'].nunique()}")
    print(f"  Backends tested: {results_df['backend'].unique()}")

    # Show a few key results
    print(f"\n{results_df.to_string(index=False)}")


if __name__ == '__main__':
    main()
