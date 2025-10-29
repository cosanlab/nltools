"""
Systematic benchmarking of ridge regression for neuroimaging workflows.

Neuroimaging Convention:
- X (features): Design matrix (time × features) - task regressors, stimuli, etc.
- y (voxels): Brain data (time × voxels) - what we're predicting

Benchmark Grid:
- Time-series length: 500 (task fMRI), 1000 (naturalistic fMRI)
- Num voxels: 50k (3mm), 230k (2mm)
- Num features: 50, 100 (typical design matrix sizes)
- Estimation style: estimates-only (fixed alpha), fit-only (5-fold CV)

Total: 2 × 2 × 2 × 2 = 16 conditions × 2 backends = 32 benchmarks

Usage:
    # Dry run with defaults
    python benchmarking.py --dry-run

    # Fast test (estimates only, small problem)
    python benchmarking.py -n 500 -v 50000 -f 50 -e estimates

    # Custom configuration
    python benchmarking.py -n 500 -v 50000 -f "10,50,100" -e fit --cv-folds 3
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
import platform
import argparse
from typing import Dict, List, Tuple

# Import nltools components
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nltools.algorithms.ridge import ridge_svd, ridge_cv
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
    cv: int = 5,
    n_alphas: int = 10
) -> Tuple[float, float]:
    """
    Benchmark ridge regression with cross-validation (fit for prediction).

    This is for when you care about out-of-sample prediction accuracy
    and want to tune hyperparameters via CV.

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
    n_alphas : int
        Number of alpha values to test (default: 10)

    Returns
    -------
    time_seconds : float
    memory_mb : float
    """
    mem_start = get_memory_mb()

    start = time.perf_counter()
    result = ridge_cv(X, y, alphas=np.logspace(-2, 2, n_alphas), cv=cv, backend=backend)
    end = time.perf_counter()

    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start

    return end - start, memory_mb


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic ridge regression benchmarks for neuroimaging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run with defaults
  %(prog)s --dry-run

  # Fast test (estimates only, one problem size)
  %(prog)s -n 500 -v 50000 -f 50 -e estimates

  # Custom CV settings
  %(prog)s --cv-folds 3 --cv-alphas 5

  # Test different feature counts
  %(prog)s -n 500 -v 50000 -f "10,50,100"

  # Full grid without GPU
  %(prog)s --no-gpu
        """
    )

    parser.add_argument(
        "-n", "--samples",
        type=str,
        default="500,1000",
        help="Time-series lengths (comma-separated, default: 500,1000)"
    )

    parser.add_argument(
        "-v", "--voxels",
        type=str,
        default="50000,230000",
        help="Voxel counts (comma-separated, default: 50000,230000)"
    )

    parser.add_argument(
        "-f", "--features",
        type=str,
        default="50,100",
        help="Feature counts for design matrix (comma-separated, default: 50,100)"
    )

    parser.add_argument(
        "-e", "--estimation",
        type=str,
        default="estimates,fit",
        help="Estimation styles: 'estimates', 'fit', or both (default: estimates,fit)"
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of CV folds for fit-only (default: 5)"
    )

    parser.add_argument(
        "--cv-alphas",
        type=int,
        default=10,
        help="Number of alpha values to test in CV (default: 10)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show benchmark plan without running"
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU benchmarks (CPU only)"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="results_ridge_systematic.csv",
        help="Output CSV filename (default: results_ridge_systematic.csv)"
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output, no progress bars"
    )

    return parser.parse_args()


def estimate_runtime(n_samples, n_voxels, n_features, estimation_style, cv_folds=5, cv_alphas=10):
    """
    Estimate runtime for a single condition.

    Parameters
    ----------
    n_samples : int
        Number of time points
    n_voxels : int
        Number of voxels (targets)
    n_features : int
        Number of features (predictors)
    estimation_style : str
        'estimates_only' or 'fit_only'
    cv_folds : int
        Number of CV folds
    cv_alphas : int
        Number of alpha values

    Returns
    -------
    numpy_time : float
        Estimated NumPy time in seconds
    torch_time : float
        Estimated PyTorch time in seconds
    """
    # Complexity scaling: Ridge via SVD is O(n_samples * n_features^2 + n_features^3)
    # For neuroimaging: n_features << n_voxels, so cost is dominated by n_features

    # Reference timing: 500 samples, 50k voxels, 50 features
    # Estimates: ~1s, Fit (5-fold CV, 10 alphas): ~120s (NumPy), ~60s (PyTorch)
    reference_samples = 500
    reference_voxels = 50000
    reference_features = 50

    # Compute scaling factors
    n_factor = n_samples / reference_samples
    v_factor = n_voxels / reference_voxels
    f_factor = n_features / reference_features  # Linear in features for small n_features

    # Base timings (reference configuration)
    if estimation_style == "estimates_only":
        numpy_base = 1.0 * n_factor * v_factor * f_factor
        torch_base = 1.0 * n_factor * v_factor * f_factor
    else:  # fit_only
        numpy_base = 120.0 * n_factor * v_factor * f_factor
        torch_base = 60.0 * n_factor * v_factor * f_factor

    # Scale for CV settings (only affects fit_only)
    if estimation_style == "fit_only":
        cv_scale = cv_folds / 5.0
        alpha_scale = cv_alphas / 10.0
        numpy_base *= cv_scale * alpha_scale
        torch_base *= cv_scale * alpha_scale

    return numpy_base, torch_base


def format_time(seconds):
    """Format seconds as human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def print_dry_run_summary(config):
    """Print formatted dry-run summary with time estimates."""
    print("=" * 80)
    print("DRY RUN: Systematic Ridge Regression Benchmarks")
    print("=" * 80)

    # Configuration summary
    print("\nConfiguration:")
    samples_str = ", ".join(str(x) for x in config['samples'])
    voxels_str = ", ".join(f"{x:,}" for x in config['voxels'])
    features_str = ", ".join(str(x) for x in config['features'])
    styles_str = ", ".join(config['estimation_styles'])

    print(f"  Samples (n_samples):   {samples_str}")
    print(f"  Voxels (n_voxels):     {voxels_str}")
    print(f"  Features (n_features): {features_str}")
    print(f"  Estimation styles:     {styles_str}")
    print(f"  CV folds:              {config['cv_folds']}")
    print(f"  Alpha grid size:       {config['cv_alphas']}")
    print(f"  GPU available:         {config['gpu_available']}")
    if config['gpu_available']:
        print(f"  GPU device:            {config['gpu_device']}")
    print(f"  Skip GPU:              {config['no_gpu']}")

    # Build condition list
    conditions = []
    for n_samples in config['samples']:
        for n_voxels in config['voxels']:
            for n_features in config['features']:
                for style in config['estimation_styles']:
                    conditions.append((n_samples, n_voxels, n_features, style))

    n_backends = 1 if config['no_gpu'] or not config['gpu_available'] else 2
    total_runs = len(conditions) * n_backends

    print(f"\n{'=' * 80}")
    print(f"Benchmark Grid ({len(conditions)} conditions × {n_backends} backend(s) = {total_runs} runs)")
    print("=" * 80)

    # Print table header
    header = f"\n{'Cond':<6} {'Samples':<8} {'Voxels':<10} {'Features':<10} {'Style':<13} {'NumPy':<10} {'PyTorch':<10} {'Total':<10}"
    print(header)
    print("-" * 90)

    # Estimate each condition
    total_time = 0
    estimates_time = 0
    fit_time = 0

    for i, (n_samples, n_voxels, n_features, style) in enumerate(conditions, 1):
        numpy_time, torch_time = estimate_runtime(
            n_samples, n_voxels, n_features, style,
            config['cv_folds'], config['cv_alphas']
        )

        if config['no_gpu'] or not config['gpu_available']:
            cond_time = numpy_time
        else:
            cond_time = numpy_time + torch_time

        total_time += cond_time
        if style == "estimates_only":
            estimates_time += cond_time
        else:
            fit_time += cond_time

        # Format row
        style_short = "estimates" if style == "estimates_only" else f"fit ({config['cv_folds']}-CV)"
        row = (f"{i}/{len(conditions):<4} {n_samples:<8} {n_voxels:<10,} {n_features:<10} {style_short:<13} "
               f"{format_time(numpy_time):<10} {format_time(torch_time):<10} {format_time(cond_time):<10}")
        print(row)

    # Summary
    print("\n" + "=" * 80)
    print(f"Total Estimated Runtime: {format_time(total_time)}")
    print("=" * 80)

    if estimates_time > 0:
        print(f"  Estimates-only subtotal: {format_time(estimates_time)}")
    if fit_time > 0:
        print(f"  Fit-only subtotal:       {format_time(fit_time)}")

    print(f"\nOutput file: {config['output']}")
    print("\nTo run: Remove --dry-run flag")
    print("=" * 80)


def run_systematic_benchmarks(config=None) -> pd.DataFrame:
    """
    Run systematic benchmark grid for neuroimaging workflows.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary with keys:
        - samples: list of n_samples values
        - voxels: list of n_voxels values
        - estimation_styles: list of style strings
        - cv_folds: int
        - cv_alphas: int
        - no_gpu: bool
        - quiet: bool

    Returns
    -------
    results : pd.DataFrame
        Columns: n_samples, n_voxels, estimation_style, backend,
                 time_seconds, memory_mb, speedup_vs_numpy
    """
    # Use default config if not provided (for backward compatibility)
    if config is None:
        config = {
            'samples': [500, 1000],
            'voxels': [50000, 230000],
            'features': [50, 100],
            'estimation_styles': ["estimates_only", "fit_only"],
            'cv_folds': 5,
            'cv_alphas': 10,
            'no_gpu': False,
            'quiet': False,
        }

    if not config.get('quiet', False):
        print("=" * 80)
        print("Systematic Ridge Regression Benchmarks for Neuroimaging")
        print("=" * 80)

    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()

    if not config.get('quiet', False):
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
        samples_str = ", ".join(str(x) for x in config['samples'])
        voxels_str = ", ".join(f"{x:,}" for x in config['voxels'])
        features_str = ", ".join(str(x) for x in config['features'])
        styles_str = ", ".join(config['estimation_styles'])
        print(f"- Samples: {samples_str}")
        print(f"- Voxels: {voxels_str}")
        print(f"- Features: {features_str}")
        print(f"- Estimation: {styles_str}")
        print(f"- CV folds: {config['cv_folds']}")
        print(f"- Alpha grid: {config['cv_alphas']}")
        print("=" * 80 + "\n")

    # Build condition list from config
    n_samples_options = [(n, f"n{n}") for n in config['samples']]
    n_voxels_options = [(v, f"v{v}") for v in config['voxels']]
    n_features_options = [(f, f"f{f}") for f in config['features']]

    # Map estimation style names
    style_map = {
        "estimates_only": "Estimates Only (fixed α=1.0)",
        "fit_only": f"Fit Only ({config['cv_folds']}-fold CV)"
    }
    estimation_styles = [
        (style, style_map.get(style, style))
        for style in config['estimation_styles']
    ]

    # Results storage
    results = []

    # Track numpy baselines for speedup calculation
    numpy_baselines = {}

    # Count total conditions
    total_conditions = len(n_samples_options) * len(n_voxels_options) * len(n_features_options) * len(estimation_styles)

    # Set up progress bar if available and not quiet
    use_progress = HAS_TQDM and not config.get('quiet', False)
    if use_progress:
        pbar = tqdm(total=total_conditions, desc="Benchmarking", unit="cond")
    else:
        pbar = None

    # Iterate through all combinations
    condition_num = 0

    for n_samples, samples_label in n_samples_options:
        for n_voxels, voxels_label in n_voxels_options:
            for n_features, features_label in n_features_options:
                for est_style, est_label in estimation_styles:
                    condition_num += 1

                    # Update progress bar description
                    if pbar:
                        pbar.set_description(f"Cond {condition_num}/{total_conditions}: n={n_samples}, v={n_voxels:,}, f={n_features}, {est_style[:3]}")

                    if not config.get('quiet', False) and not pbar:
                        print(f"\n{'='*80}")
                        print(f"Condition {condition_num}/{total_conditions}")
                        print(f"{'='*80}")
                        print(f"  Samples: {n_samples} ({samples_label})")
                        print(f"  Voxels: {n_voxels:,} ({voxels_label})")
                        print(f"  Features: {n_features} ({features_label})")
                        print(f"  Style: {est_label}")
                        print(f"{'-'*80}")

                    # Generate data (neuroimaging convention: X = design matrix, y = brain data)
                    if not config.get('quiet', False) and not pbar:
                        print(f"  Generating data: X={n_samples}×{n_features}, y={n_samples}×{n_voxels:,}...")
                    X = np.random.randn(n_samples, n_features).astype(np.float32)
                    y = np.random.randn(n_samples, n_voxels).astype(np.float32)

                    # Create condition key for baseline tracking
                    condition_key = f"{samples_label}_{voxels_label}_{features_label}_{est_style}"

                # Benchmark NumPy
                if not config.get('quiet', False) and not pbar:
                    print(f"  Testing NumPy backend...", end=" ", flush=True)
                backend_np = Backend('numpy')

                if est_style == "estimates_only":
                    time_np, mem_np = benchmark_estimates_only(X, y, backend_np, alpha=1.0)
                else:  # fit_only
                    time_np, mem_np = benchmark_fit_only(X, y, backend_np, cv=config['cv_folds'], n_alphas=config['cv_alphas'])

                if not config.get('quiet', False) and not pbar:
                    print(f"{time_np:.2f}s (memory: {mem_np:+.1f} MB)")

                results.append({
                    'n_samples': n_samples,
                    'samples_label': samples_label,
                    'n_voxels': n_voxels,
                    'voxels_label': voxels_label,
                    'n_features': n_features,
                    'features_label': features_label,
                    'estimation_style': est_style,
                    'backend': 'numpy',
                    'time_seconds': time_np,
                    'memory_mb': mem_np,
                    'speedup_vs_numpy': 1.0
                })

                numpy_baselines[condition_key] = time_np

                # Benchmark PyTorch (if available and not disabled)
                skip_gpu = config.get('no_gpu', False) or not gpu_available
                if not skip_gpu:
                    if not config.get('quiet', False) and not pbar:
                        print(f"  Testing PyTorch backend...", end=" ", flush=True)
                    backend_torch = Backend('torch')

                    if est_style == "estimates_only":
                        time_torch, mem_torch = benchmark_estimates_only(X, y, backend_torch, alpha=1.0)
                    else:  # fit_only
                        time_torch, mem_torch = benchmark_fit_only(X, y, backend_torch, cv=config['cv_folds'], n_alphas=config['cv_alphas'])

                    speedup = time_np / time_torch
                    if not config.get('quiet', False) and not pbar:
                        print(f"{time_torch:.2f}s (memory: {mem_torch:+.1f} MB, speedup: {speedup:.2f}x)")

                    results.append({
                        'n_samples': n_samples,
                        'samples_label': samples_label,
                        'n_voxels': n_voxels,
                        'voxels_label': voxels_label,
                        'n_features': n_features,
                        'features_label': features_label,
                        'estimation_style': est_style,
                        'backend': backend_torch.name,
                        'time_seconds': time_torch,
                        'memory_mb': mem_torch,
                        'speedup_vs_numpy': speedup
                    })
                elif not config.get('quiet', False) and not pbar:
                    if config.get('no_gpu', False):
                        print(f"  Skipping PyTorch backend (--no-gpu flag)")
                    else:
                        print(f"  Skipping PyTorch backend (GPU not available)")

                # Update progress bar
                if pbar:
                    pbar.update(1)

                # Clean up to free memory
                del X, y

    # Close progress bar
    if pbar:
        pbar.close()

    if not config.get('quiet', False):
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
    """Run systematic benchmarks with CLI support and save results."""
    # Parse command-line arguments
    args = parse_args()

    # Parse comma-separated arguments
    try:
        samples = [int(x.strip()) for x in args.samples.split(',')]
    except ValueError:
        print(f"Error: Invalid samples argument '{args.samples}'. Must be comma-separated integers.")
        return

    try:
        voxels = [int(x.strip()) for x in args.voxels.split(',')]
    except ValueError:
        print(f"Error: Invalid voxels argument '{args.voxels}'. Must be comma-separated integers.")
        return

    try:
        features = [int(x.strip()) for x in args.features.split(',')]
    except ValueError:
        print(f"Error: Invalid features argument '{args.features}'. Must be comma-separated integers.")
        return

    # Parse estimation styles
    estimation_map = {
        'estimates': 'estimates_only',
        'fit': 'fit_only',
        'estimates_only': 'estimates_only',
        'fit_only': 'fit_only'
    }
    try:
        styles = []
        for x in args.estimation.split(','):
            style = x.strip().lower()
            if style not in estimation_map:
                print(f"Error: Invalid estimation style '{style}'. Must be 'estimates' or 'fit'.")
                return
            styles.append(estimation_map[style])
    except Exception as e:
        print(f"Error parsing estimation argument: {e}")
        return

    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()

    # Build configuration
    config = {
        'samples': samples,
        'voxels': voxels,
        'features': features,
        'estimation_styles': styles,
        'cv_folds': args.cv_folds,
        'cv_alphas': args.cv_alphas,
        'no_gpu': args.no_gpu,
        'quiet': args.quiet,
        'output': args.output,
        'gpu_available': gpu_available,
        'gpu_device': gpu_info['device'] if gpu_available else 'none',
    }

    # Handle dry-run
    if args.dry_run:
        print_dry_run_summary(config)
        return

    # Run benchmarks
    results_df = run_systematic_benchmarks(config)

    # Save to CSV
    output_path = os.path.join(
        os.path.dirname(__file__),
        args.output
    )
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    # Print summary
    if not args.quiet:
        print_summary(results_df)

        # Show full results table
        print(f"\n{'='*80}")
        print("FULL RESULTS TABLE")
        print(f"{'='*80}\n")
        print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
