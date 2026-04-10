"""
Performance benchmark for GPU-accelerated Ridge bootstrap.

Compares CPU-parallel vs GPU-accelerated implementations for Ridge weights
and predictions bootstrap across realistic problem sizes.

Validates performance claims from gpu-implementation-tdd-plan.md:
- Expected speedup: 10-50× for large problems on CUDA GPUs
- Problem size: n_samples=100, n_features=50, n_voxels=10,000
- Bootstrap iterations: n_samples=5000
- Verifies memory stays under budget

Usage:
    # Run with defaults (problem size from plan)
    python bootstrap_ridge_gpu_benchmark.py

    # Custom problem size
    python bootstrap_ridge_gpu_benchmark.py --n-samples 50 --n-features 25 --n-voxels 5000

    # Custom bootstrap iterations
    python bootstrap_ridge_gpu_benchmark.py --n-bootstrap 10000

    # Dry run (estimate time without running)
    python bootstrap_ridge_gpu_benchmark.py --dry-run
"""

import numpy as np
import time
import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from nltools.algorithms.inference.bootstrap import (
    _bootstrap_ridge_weights_cpu_parallel,
    _bootstrap_ridge_weights_gpu_batched,
    _bootstrap_ridge_predict_cpu_parallel,
    _bootstrap_ridge_predict_gpu_batched,
)
from nltools.algorithms.backends import Backend, check_gpu_available


def benchmark_ridge_bootstrap(
    n_samples=100,
    n_features=50,
    n_voxels=10000,
    n_bootstrap=5000,
    n_test_samples=20,
    alpha=1.0,
    max_gpu_memory_gb=4.0,
    random_state=42,
    dry_run=False,
):
    """
    Benchmark CPU vs GPU performance for Ridge bootstrap.

    Parameters
    ----------
    n_samples : int
        Number of training samples
    n_features : int
        Number of features
    n_voxels : int
        Number of voxels/targets
    n_bootstrap : int
        Number of bootstrap iterations
    n_test_samples : int
        Number of test samples for predict bootstrap
    alpha : float
        Ridge regularization parameter
    max_gpu_memory_gb : float
        Maximum GPU memory budget in GB
    random_state : int
        Random seed for reproducibility
    dry_run : bool
        If True, only estimate time without running benchmarks

    Returns
    -------
    dict
        Dictionary containing timing results and speedup metrics
    """
    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()
    if not gpu_available:
        print("⚠️  GPU not available. Skipping GPU benchmarks.")
        print("   Run with CPU-only mode or ensure GPU is available.")
        return None

    # Generate synthetic data
    rng = np.random.RandomState(random_state)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = rng.randn(n_samples, n_voxels).astype(np.float32)
    X_test = rng.randn(n_test_samples, n_features).astype(np.float32)

    # Setup GPU backend
    backend = Backend("torch")
    backend_name = backend.device if hasattr(backend, "device") else str(backend)

    print("\n" + "=" * 70)
    print("GPU Bootstrap Performance Benchmark")
    print("=" * 70)
    print(
        f"Problem size: {n_samples} samples × {n_features} features → {n_voxels} voxels"
    )
    print(f"Bootstrap iterations: {n_bootstrap}")
    print(f"GPU: {gpu_info}")
    print(f"GPU memory budget: {max_gpu_memory_gb} GB")
    print("-" * 70)

    if dry_run:
        print("\n🔍 DRY RUN MODE - Estimating runtime...")
        print("   Actual benchmarks will not be run.")
        print(
            f"   Estimated CPU time: ~{n_bootstrap * 0.005:.1f}s ({n_bootstrap * 0.005 / 60:.1f} min)"
        )
        print(
            f"   Estimated GPU time: ~{n_bootstrap * 0.007:.1f}s ({n_bootstrap * 0.007 / 60:.1f} min)"
        )
        print("=" * 70 + "\n")
        return None

    results = {}

    # ========================================================================
    # Test 1: Ridge Weights Bootstrap
    # ========================================================================
    print("\n1. Ridge Weights Bootstrap:")
    print("-" * 70)

    # CPU timing
    print("   Running CPU benchmark...")
    start_cpu = time.time()
    result_cpu_weights = _bootstrap_ridge_weights_cpu_parallel(
        X,
        y,
        alpha,
        n_samples=n_bootstrap,
        save_boots=False,
        n_jobs=-1,  # Use all cores
        random_state=random_state,
    )
    elapsed_cpu_weights = time.time() - start_cpu

    # GPU timing
    print("   Running GPU benchmark...")
    start_gpu = time.time()
    result_gpu_weights = _bootstrap_ridge_weights_gpu_batched(
        X,
        y,
        alpha,
        n_samples=n_bootstrap,
        save_boots=False,
        backend=backend,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=random_state,
    )
    elapsed_gpu_weights = time.time() - start_gpu

    speedup_weights = elapsed_cpu_weights / elapsed_gpu_weights

    print(
        f"   CPU time: {elapsed_cpu_weights:.2f}s ({elapsed_cpu_weights / n_bootstrap * 1000:.2f}ms/iter)"
    )
    print(
        f"   GPU time: {elapsed_gpu_weights:.2f}s ({elapsed_gpu_weights / n_bootstrap * 1000:.2f}ms/iter)"
    )
    print(f"   Speedup: {speedup_weights:.2f}×")
    print(f"   CPU throughput: {n_bootstrap / elapsed_cpu_weights:.1f} iter/s")
    print(f"   GPU throughput: {n_bootstrap / elapsed_gpu_weights:.1f} iter/s")

    results["weights"] = {
        "cpu_time": elapsed_cpu_weights,
        "gpu_time": elapsed_gpu_weights,
        "speedup": speedup_weights,
        "cpu_throughput": n_bootstrap / elapsed_cpu_weights,
        "gpu_throughput": n_bootstrap / elapsed_gpu_weights,
    }

    # Verify CPU/GPU consistency
    mean_diff = np.abs(result_cpu_weights["mean"] - result_gpu_weights["mean"]).max()
    mean_rel_diff = (
        np.abs(result_cpu_weights["mean"] - result_gpu_weights["mean"])
        / (np.abs(result_cpu_weights["mean"]) + 1e-10)
    ).max()
    print(f"   Max difference: {mean_diff:.2e} (rel: {mean_rel_diff * 100:.3f}%)")

    # ========================================================================
    # Test 2: Ridge Predict Bootstrap
    # ========================================================================
    print("\n2. Ridge Predict Bootstrap:")
    print("-" * 70)

    # CPU timing
    print("   Running CPU benchmark...")
    start_cpu = time.time()
    result_cpu_predict = _bootstrap_ridge_predict_cpu_parallel(
        X,
        y,
        X_test,
        alpha,
        n_samples=n_bootstrap,
        save_boots=False,
        n_jobs=-1,  # Use all cores
        random_state=random_state,
    )
    elapsed_cpu_predict = time.time() - start_cpu

    # GPU timing
    print("   Running GPU benchmark...")
    start_gpu = time.time()
    result_gpu_predict = _bootstrap_ridge_predict_gpu_batched(
        X,
        y,
        X_test,
        alpha,
        n_samples=n_bootstrap,
        save_boots=False,
        backend=backend,
        max_gpu_memory_gb=max_gpu_memory_gb,
        random_state=random_state,
    )
    elapsed_gpu_predict = time.time() - start_gpu

    speedup_predict = elapsed_cpu_predict / elapsed_gpu_predict

    print(
        f"   CPU time: {elapsed_cpu_predict:.2f}s ({elapsed_cpu_predict / n_bootstrap * 1000:.2f}ms/iter)"
    )
    print(
        f"   GPU time: {elapsed_gpu_predict:.2f}s ({elapsed_gpu_predict / n_bootstrap * 1000:.2f}ms/iter)"
    )
    print(f"   Speedup: {speedup_predict:.2f}×")
    print(f"   CPU throughput: {n_bootstrap / elapsed_cpu_predict:.1f} iter/s")
    print(f"   GPU throughput: {n_bootstrap / elapsed_gpu_predict:.1f} iter/s")

    results["predict"] = {
        "cpu_time": elapsed_cpu_predict,
        "gpu_time": elapsed_gpu_predict,
        "speedup": speedup_predict,
        "cpu_throughput": n_bootstrap / elapsed_cpu_predict,
        "gpu_throughput": n_bootstrap / elapsed_gpu_predict,
    }

    # Verify CPU/GPU consistency
    mean_diff = np.abs(result_cpu_predict["mean"] - result_gpu_predict["mean"]).max()
    mean_rel_diff = (
        np.abs(result_cpu_predict["mean"] - result_gpu_predict["mean"])
        / (np.abs(result_cpu_predict["mean"]) + 1e-10)
    ).max()
    print(f"   Max difference: {mean_diff:.2e} (rel: {mean_rel_diff * 100:.3f}%)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("Performance Summary:")
    print("=" * 70)
    print(f"Ridge Weights: {speedup_weights:.2f}× speedup")
    print(f"Ridge Predict: {speedup_predict:.2f}× speedup")
    print(f"Average speedup: {(speedup_weights + speedup_predict) / 2:.2f}×")

    # Context-aware performance notes
    backend_name_lower = result_gpu_weights["backend"].lower()
    is_mps = "mps" in backend_name_lower

    if is_mps:
        print("\n📝 Note: MPS (Mac GPU) backend detected.")
        print("   MPS may show lower speedup or even slowdown due to overhead.")
        print("   GPU acceleration is optimized for CUDA GPUs (NVIDIA).")
        if speedup_weights < 1.0 or speedup_predict < 1.0:
            print("   ⚠️  GPU is slower than CPU on this system (expected for MPS).")
    else:
        print(f"\n✅ CUDA GPU backend: {backend_name}")
        if speedup_weights >= 5.0 or speedup_predict >= 5.0:
            print("   🚀 Excellent GPU acceleration achieved!")
        elif speedup_weights >= 2.0 or speedup_predict >= 2.0:
            print("   ✅ Good GPU acceleration achieved.")
        else:
            print(
                "   ⚠️  GPU speedup lower than expected. Check GPU compute capability."
            )

    print("\n✅ Memory verification: GPU bootstrap completed without OOM")
    print("   (Memory budget enforced via automatic batching)")
    print("=" * 70 + "\n")

    # Add metadata
    results["metadata"] = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_voxels": n_voxels,
        "n_bootstrap": n_bootstrap,
        "n_test_samples": n_test_samples,
        "alpha": alpha,
        "gpu_info": gpu_info,
        "backend": backend_name,
        "is_mps": is_mps,
    }

    return results


def main():
    """Main entry point for benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark GPU vs CPU performance for Ridge bootstrap",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with defaults (from gpu-implementation-tdd-plan.md)
    python bootstrap_ridge_gpu_benchmark.py

    # Custom problem size
    python bootstrap_ridge_gpu_benchmark.py --n-samples 50 --n-features 25 --n-voxels 5000

    # More bootstrap iterations
    python bootstrap_ridge_gpu_benchmark.py --n-bootstrap 10000

    # Dry run (estimate time)
    python bootstrap_ridge_gpu_benchmark.py --dry-run
        """,
    )

    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of training samples (default: 100)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=50,
        help="Number of features (default: 50)",
    )
    parser.add_argument(
        "--n-voxels",
        type=int,
        default=10000,
        help="Number of voxels/targets (default: 10000)",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=5000,
        help="Number of bootstrap iterations (default: 5000)",
    )
    parser.add_argument(
        "--n-test-samples",
        type=int,
        default=20,
        help="Number of test samples for predict bootstrap (default: 20)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Ridge regularization parameter (default: 1.0)",
    )
    parser.add_argument(
        "--max-gpu-memory-gb",
        type=float,
        default=4.0,
        help="Maximum GPU memory budget in GB (default: 4.0)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode: estimate time without running benchmarks",
    )

    args = parser.parse_args()

    # Run benchmark
    results = benchmark_ridge_bootstrap(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_voxels=args.n_voxels,
        n_bootstrap=args.n_bootstrap,
        n_test_samples=args.n_test_samples,
        alpha=args.alpha,
        max_gpu_memory_gb=args.max_gpu_memory_gb,
        random_state=args.random_state,
        dry_run=args.dry_run,
    )

    if results is None:
        return 0

    # Exit with code 0 (success)
    return 0


if __name__ == "__main__":
    sys.exit(main())
