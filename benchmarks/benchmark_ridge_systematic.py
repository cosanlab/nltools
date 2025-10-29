"""
Systematic benchmarking of ridge regression for neuroimaging workflows.

Benchmark Grid:
- Time-series length: 500 (task fMRI), 1000 (naturalistic fMRI)
- Num voxels: 50k (3mm), 230k (2mm)
- Estimation style: estimates-only (fixed alpha), fit-only (5-fold CV)

Total: 2 × 2 × 2 = 8 conditions × 2 backends = 16 benchmarks
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
import platform
from typing import Dict, List, Tuple

# Import nltools components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nltools.algorithms.ridge import ridge_svd, ridge_cv
from nltools.backends import Backend, check_gpu_available


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def benchmark_estimates_only(
    X: np.ndarray,
    y: np.ndarray,
    backend: Backend,
    alpha: float = 1.0
) -> Tuple[float, float]:
    """
    Benchmark ridge regression with fixed alpha (no CV).

    This is for when you only care about coefficient estimates,
    not prediction accuracy or hyperparameter tuning.

    Returns
    -------
    time_seconds : float
    memory_mb : float
    """
    mem_start = get_memory_mb()

    start = time.perf_counter()
    coef = ridge_svd(X, y, alpha=alpha, backend=backend)
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def benchmark_fit_only(
    X: np.ndarray,
    y: np.ndarray,
    backend: Backend,
    cv: int = 5
) -> Tuple[float, float]:
    """
    Benchmark ridge regression with cross-validation (fit for prediction).

    This is for when you care about out-of-sample prediction accuracy
    and want to tune hyperparameters via CV.

    Returns
    -------
    time_seconds : float
    memory_mb : float
    """
    mem_start = get_memory_mb()

    start = time.perf_counter()
    result = ridge_cv(X, y, alphas=np.logspace(-2, 2, 10), cv=cv, backend=backend)
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def run_systematic_benchmarks() -> pd.DataFrame:
    """
    Run systematic benchmark grid for neuroimaging workflows.

    Returns
    -------
    results : pd.DataFrame
        Columns: n_samples, n_voxels, estimation_style, backend,
                 time_seconds, memory_mb, speedup_vs_numpy
    """
    print("=" * 80)
    print("Systematic Ridge Regression Benchmarks for Neuroimaging")
    print("=" * 80)

    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()
    print(f"\nGPU Available: {gpu_available}")
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
    print("- Time-series length: 500 (task fMRI), 1000 (naturalistic fMRI)")
    print("- Num voxels: 50k (3mm), 230k (2mm)")
    print("- Estimation: estimates-only (fixed α), fit-only (5-fold CV)")
    print("=" * 80 + "\n")

    # Define benchmark grid
    n_samples_options = [
        (500, "task_fmri"),
        (1000, "naturalistic_fmri")
    ]

    n_voxels_options = [
        (50000, "3mm_resolution"),
        (230000, "2mm_resolution")
    ]

    estimation_styles = [
        ("estimates_only", "Estimates Only (fixed α=1.0)"),
        ("fit_only", "Fit Only (5-fold CV)")
    ]

    # Results storage
    results = []

    # Track numpy baselines for speedup calculation
    numpy_baselines = {}

    # Iterate through all combinations
    condition_num = 0
    total_conditions = len(n_samples_options) * len(n_voxels_options) * len(estimation_styles)

    for n_samples, samples_label in n_samples_options:
        for n_voxels, voxels_label in n_voxels_options:
            for est_style, est_label in estimation_styles:
                condition_num += 1

                print(f"\n{'='*80}")
                print(f"Condition {condition_num}/{total_conditions}")
                print(f"{'='*80}")
                print(f"  Samples: {n_samples} ({samples_label})")
                print(f"  Voxels: {n_voxels:,} ({voxels_label})")
                print(f"  Style: {est_label}")
                print(f"{'-'*80}")

                # Generate data
                print(f"  Generating data: {n_samples}×{n_voxels:,} = {n_samples*n_voxels:,} elements...")
                X = np.random.randn(n_samples, n_voxels).astype(np.float32)
                y = np.random.randn(n_samples).astype(np.float32)

                # Create condition key for baseline tracking
                condition_key = f"{samples_label}_{voxels_label}_{est_style}"

                # Benchmark NumPy
                print(f"  Testing NumPy backend...", end=" ", flush=True)
                backend_np = Backend('numpy')

                if est_style == "estimates_only":
                    time_np, mem_np = benchmark_estimates_only(X, y, backend_np, alpha=1.0)
                else:  # fit_only
                    time_np, mem_np = benchmark_fit_only(X, y, backend_np, cv=5)

                print(f"{time_np:.2f}s (memory: {mem_np:+.1f} MB)")

                results.append({
                    'n_samples': n_samples,
                    'samples_label': samples_label,
                    'n_voxels': n_voxels,
                    'voxels_label': voxels_label,
                    'estimation_style': est_style,
                    'backend': 'numpy',
                    'time_seconds': time_np,
                    'memory_mb': mem_np,
                    'speedup_vs_numpy': 1.0
                })

                numpy_baselines[condition_key] = time_np

                # Benchmark PyTorch (if available)
                if gpu_available:
                    print(f"  Testing PyTorch backend...", end=" ", flush=True)
                    backend_torch = Backend('torch')

                    if est_style == "estimates_only":
                        time_torch, mem_torch = benchmark_estimates_only(X, y, backend_torch, alpha=1.0)
                    else:  # fit_only
                        time_torch, mem_torch = benchmark_fit_only(X, y, backend_torch, cv=5)

                    speedup = time_np / time_torch
                    print(f"{time_torch:.2f}s (memory: {mem_torch:+.1f} MB, speedup: {speedup:.2f}x)")

                    results.append({
                        'n_samples': n_samples,
                        'samples_label': samples_label,
                        'n_voxels': n_voxels,
                        'voxels_label': voxels_label,
                        'estimation_style': est_style,
                        'backend': backend_torch.name,
                        'time_seconds': time_torch,
                        'memory_mb': mem_torch,
                        'speedup_vs_numpy': speedup
                    })
                else:
                    print(f"  Skipping PyTorch backend (GPU not available)")

                # Clean up to free memory
                del X, y

    print(f"\n{'='*80}")
    print("Benchmark Complete!")
    print(f"{'='*80}")

    return pd.DataFrame(results)


def print_summary(df: pd.DataFrame):
    """Print formatted summary of benchmark results."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLES")
    print(f"{'='*80}\n")

    # Group by conditions and show speedup
    print("Speedup Summary (PyTorch vs NumPy):")
    print(f"{'-'*80}")

    # Get unique conditions
    conditions = df[['n_samples', 'n_voxels', 'estimation_style']].drop_duplicates()

    for _, cond in conditions.iterrows():
        subset = df[
            (df['n_samples'] == cond['n_samples']) &
            (df['n_voxels'] == cond['n_voxels']) &
            (df['estimation_style'] == cond['estimation_style'])
        ]

        np_row = subset[subset['backend'] == 'numpy'].iloc[0]
        torch_rows = subset[subset['backend'] != 'numpy']

        if len(torch_rows) > 0:
            torch_row = torch_rows.iloc[0]
            speedup = torch_row['speedup_vs_numpy']

            est_label = "Est" if cond['estimation_style'] == 'estimates_only' else "CV"
            print(f"n={cond['n_samples']:4d}, v={cond['n_voxels']:6d}, {est_label}: "
                  f"NumPy={np_row['time_seconds']:6.2f}s, "
                  f"Torch={torch_row['time_seconds']:6.2f}s, "
                  f"Speedup={speedup:.2f}x")

    print(f"\n{'-'*80}")
    print("Key Findings:")

    # Best speedups
    torch_results = df[df['backend'] != 'numpy']
    if len(torch_results) > 0:
        best_speedup = torch_results.loc[torch_results['speedup_vs_numpy'].idxmax()]
        worst_speedup = torch_results.loc[torch_results['speedup_vs_numpy'].idxmin()]

        print(f"  Best speedup: {best_speedup['speedup_vs_numpy']:.2f}x")
        print(f"    (n={best_speedup['n_samples']}, v={best_speedup['n_voxels']}, "
              f"{best_speedup['estimation_style']})")
        print(f"  Worst speedup: {worst_speedup['speedup_vs_numpy']:.2f}x")
        print(f"    (n={worst_speedup['n_samples']}, v={worst_speedup['n_voxels']}, "
              f"{worst_speedup['estimation_style']})")

        # Average speedup
        avg_speedup = torch_results['speedup_vs_numpy'].mean()
        print(f"  Average speedup: {avg_speedup:.2f}x")


def main():
    """Run systematic benchmarks and save results."""
    # Run benchmark suite
    results_df = run_systematic_benchmarks()

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(__file__),
        'results_ridge_systematic.csv'
    )
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    print_summary(results_df)

    # Show full results table
    print(f"\n{'='*80}")
    print("FULL RESULTS TABLE")
    print(f"{'='*80}\n")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
