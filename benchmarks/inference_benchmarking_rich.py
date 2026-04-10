"""
Systematic benchmarking of inference algorithms for neuroimaging workflows.

Enhanced with Rich for beautiful terminal output, interactive configuration,
and clear result visualization.

Compares CPU (NumPy), CPU-parallel (joblib), and GPU (PyTorch) implementations
across realistic problem sizes and permutation counts.

Usage:
    # Interactive mode (prompts for configuration)
    python inference_benchmarking_rich.py

    # CLI mode with defaults
    python inference_benchmarking_rich.py --algorithm one_sample --n-features "1,100"
"""

import numpy as np
import pandas as pd
import time
import psutil
import os
import platform
import argparse
import gc
from typing import Tuple, Dict, Optional, List
from itertools import product
from datetime import datetime

# Disable tqdm progress bars globally to avoid interference with Rich
# Create a no-op tqdm wrapper BEFORE importing inference modules
# This ensures tqdm calls inside inference functions don't show progress bars
import sys
import types


# Create a no-op wrapper that acts like tqdm but doesn't print anything
class NoOpTqdm:
    """No-op tqdm replacement that prevents progress bar output."""

    def __init__(self, iterable=None, *args, **kwargs):
        # Handle both iterable and non-iterable usage
        if iterable is not None:
            self.iterable = iterable
        else:
            # If no iterable, create a range based on total
            total = kwargs.get("total", 0)
            self.iterable = range(total) if total > 0 else []
        self.total = kwargs.get(
            "total", len(self.iterable) if hasattr(self.iterable, "__len__") else None
        )
        self.desc = kwargs.get("desc", "")
        self.unit = kwargs.get("unit", "it")
        self.n = 0
        self.disable = True  # Always disabled

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(self.iterable)

    def __next__(self):
        return next(iter(self.iterable))

    def update(self, n=1):
        self.n += n

    def close(self):
        pass

    def __call__(self, *args, **kwargs):
        # Handle case where tqdm is called as a function
        return self.__class__(*args, **kwargs)


# Pre-patch tqdm module BEFORE any imports that might use it
# This works even when tqdm is imported inside functions
tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = NoOpTqdm
tqdm_module.__all__ = ["tqdm"]
# Also create tqdm.auto submodule properly
auto_module = types.ModuleType("auto")
auto_module.tqdm = NoOpTqdm
auto_module.__all__ = ["tqdm"]
tqdm_module.auto = auto_module
sys.modules["tqdm"] = tqdm_module
sys.modules["tqdm.auto"] = auto_module

# Rich imports
# noqa: E402 - imports must come after tqdm patching
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.progress import (  # noqa: E402
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, Prompt  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.text import Text  # noqa: E402
from rich import box  # noqa: E402

# Import nltools components
# noqa: E402 - imports must come after tqdm patching
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from nltools.algorithms.inference import (  # noqa: E402
    one_sample_permutation_test,
    two_sample_permutation_test,
    correlation_permutation_test,
    timeseries_correlation_permutation_test,
    matrix_permutation_test,
    isc_permutation_test,
    isc_group_permutation_test,
)
from nltools.algorithms.backends import check_gpu_available  # noqa: E402

# Also patch after import in case modules were already loaded
try:
    import tqdm

    tqdm.tqdm = NoOpTqdm
    if hasattr(tqdm, "auto"):
        tqdm.auto.tqdm = NoOpTqdm
    # Ensure tqdm.auto module exists in sys.modules
    if "tqdm.auto" not in sys.modules:
        sys.modules["tqdm.auto"] = tqdm.auto if hasattr(tqdm, "auto") else auto_module
except (ImportError, AttributeError):
    pass

# Initialize Rich console
console = Console()


def get_memory_mb() -> float:
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


# ============================================================================
# Data Generation Functions (same as original)
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
    return rng.randn(n_timepoints, n_subjects, n_voxels).astype(np.float32)


# ============================================================================
# Benchmark Functions (same as original)
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
    method: str = "bootstrap",
    summary_statistic: str = "leave-one-out",
) -> Tuple[float, float]:
    """Benchmark isc_permutation_test."""
    mem_start = get_memory_mb()
    start = time.perf_counter()
    _ = isc_permutation_test(
        data,
        n_permute=n_permute,
        method=method,
        summary_statistic=summary_statistic,
        parallel=parallel,
        n_jobs=n_jobs,
        random_state=random_state,
        progress_bar=False,  # Disable progress bar to avoid conflicts
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
    method: str = "permute",
    summary_statistic: str = "pairwise",
) -> Tuple[float, float]:
    """Benchmark isc_group_permutation_test."""
    mem_start = get_memory_mb()
    start = time.perf_counter()
    _ = isc_group_permutation_test(
        group1,
        group2,
        n_permute=n_permute,
        method=method,
        summary_statistic=summary_statistic,
        parallel=parallel,
        n_jobs=n_jobs,
        random_state=random_state,
        progress_bar=False,  # Disable progress bar to avoid conflicts
    )
    end = time.perf_counter()
    mem_end = get_memory_mb()
    memory_mb = mem_end - mem_start
    return end - start, memory_mb


# ============================================================================
# Rich Display Functions
# ============================================================================


def get_device_label(device: str, n_jobs: Optional[int] = None) -> str:
    """Get human-readable label for device matching parallel kwarg semantics."""
    if device == "numpy":
        return "CPU Single (parallel=None)"
    elif device == "cpu-parallel":
        if n_jobs is None or n_jobs == -1:
            n_jobs = os.cpu_count() or 1
        return f"CPU Parallel (parallel='cpu', n_jobs={n_jobs})"
    elif device == "torch":
        return "GPU (parallel='gpu')"
    else:
        return device


def get_algorithm_label(algorithm: str, method: Optional[str] = None) -> str:
    """Get human-readable label for algorithm."""
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
    if method:
        method_labels = {
            "circle_shift": "Permutation (Circle Shift)",
            "phase_randomize": "Permutation (Phase Randomize)",
            "bootstrap": "Bootstrap",
            "permute": "Permutation",
        }
        method_label = method_labels.get(method, method.replace("_", " ").title())
        if algorithm == "timeseries_correlation":
            return f"{base_label} ({method_label})"
        elif algorithm == "isc":
            return f"{base_label} ({method_label})"
        elif algorithm == "isc_group":
            return f"{base_label} ({method_label})"
    return base_label


def format_system_info_line(config: Dict, gpu_available: bool, gpu_info: Dict) -> str:
    """Format system information as a single condensed line."""
    import torch

    parts = []
    parts.append(f"Python {platform.python_version()}")
    parts.append(f"NumPy {np.__version__}")
    parts.append(f"PyTorch {torch.__version__}")

    if gpu_available:
        parts.append(
            f"GPU: {gpu_info.get('device_name', gpu_info.get('device', 'Unknown'))}"
        )
    else:
        parts.append("GPU: None")

    parts.append(f"Algorithms: {', '.join(config['algorithms'])}")

    return " | ".join(parts)


def create_results_table(device_results: List[Dict]) -> Table:
    """Create results table from device results."""
    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("Algorithm", style="cyan", no_wrap=True)
    table.add_column("Backend", style="magenta")
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("Memory (MB)", justify="right", style="yellow")
    table.add_column("Speedup vs NumPy", justify="right", style="blue")
    table.add_column("Speedup vs CPU-Parallel", justify="right", style="blue")

    if not device_results:
        table.add_row("[dim]No results yet...[/dim]", "", "", "", "", "")
        return table

    df_device = pd.DataFrame(device_results)
    for _, row in df_device.iterrows():
        backend_style = {
            "numpy": "[dim]CPU Single[/dim]",
            "cpu-parallel": "[cyan]CPU Parallel[/cyan]",
            "torch": "[bold red]GPU[/bold red]",
        }
        backend_display = backend_style.get(row["backend"], row["backend"])

        speedup_numpy = (
            f"{row['speedup_vs_numpy']:.4f}x"
            if pd.notna(row.get("speedup_vs_numpy"))
            else "[dim]—[/dim]"
        )
        speedup_cp = (
            f"{row['speedup_vs_cpu_parallel']:.4f}x"
            if pd.notna(row.get("speedup_vs_cpu_parallel"))
            else "[dim]—[/dim]"
        )

        table.add_row(
            row["algorithm"],
            backend_display,
            f"{row['time_seconds']:.4f}",
            f"{row['memory_mb']:.4f}",
            speedup_numpy,
            speedup_cp,
        )

    return table


def print_system_info(config: Dict, gpu_available: bool, gpu_info: Dict):
    """Print system information in a styled panel (legacy function for compatibility)."""
    system_info_line = format_system_info_line(config, gpu_available, gpu_info)
    console.print(f"[dim]{system_info_line}[/dim]")


def print_benchmark_config(config: Dict):
    """Print benchmark configuration in a styled panel (legacy function for compatibility)."""
    config_table = Table.grid(padding=1)
    config_table.add_column(style="cyan", justify="right")
    config_table.add_column(style="white")

    config_table.add_row("Algorithms:", ", ".join(config["algorithms"]))
    config_table.add_row("Samples:", ", ".join(str(x) for x in config["n_samples"]))
    config_table.add_row("Features:", ", ".join(str(x) for x in config["n_features"]))
    config_table.add_row(
        "Permutations:", ", ".join(str(x) for x in config["n_permute"])
    )
    if "n_timepoints" in config:
        config_table.add_row(
            "Timepoints:", ", ".join(str(x) for x in config["n_timepoints"])
        )
    config_table.add_row(
        "Skip GPU:", "✓ Yes" if config.get("no_gpu", False) else "✗ No"
    )

    console.print(
        Panel(
            config_table,
            title="[bold green]Benchmark Configuration",
            border_style="green",
        )
    )


def print_test_summary_table(
    device_results: List[Dict], device: str, n_jobs: Optional[int] = None
):
    """Print device summary as a Rich table."""
    if not device_results:
        return

    table = Table(title=f"{get_device_label(device, n_jobs)} Summary", box=box.ROUNDED)
    table.add_column("Algorithm", style="cyan", no_wrap=True)
    table.add_column("Tests", justify="right", style="magenta")
    table.add_column("Avg Time (s)", justify="right", style="green")
    table.add_column("Avg Memory (MB)", justify="right", style="yellow")

    df_device = pd.DataFrame(device_results)
    for algorithm in df_device["algorithm"].unique():
        alg_df = df_device[df_device["algorithm"] == algorithm]
        avg_time = alg_df["time_seconds"].mean()
        avg_mem = alg_df["memory_mb"].mean()
        n_tests = len(alg_df)

        table.add_row(algorithm, str(n_tests), f"{avg_time:.4f}", f"{avg_mem:.4f}")

    console.print("\n")
    console.print(table)


def print_results_summary_table(df: pd.DataFrame):
    """Print final summary table with speedups."""
    summary_table = Table(
        title="[bold]Benchmark Summary", box=box.ROUNDED, show_header=True
    )
    summary_table.add_column("Algorithm", style="cyan", no_wrap=True)
    summary_table.add_column("Backend", style="magenta")
    summary_table.add_column("Speedup vs NumPy", justify="right", style="green")
    summary_table.add_column("Speedup vs CPU-Parallel", justify="right", style="yellow")

    for algorithm in df["algorithm"].unique():
        alg_df = df[df["algorithm"] == algorithm]

        # CPU-parallel row
        cpu_parallel_df = alg_df[alg_df["backend"] == "cpu-parallel"]
        if len(cpu_parallel_df) > 0:
            speedup_cp_numpy = cpu_parallel_df["speedup_vs_numpy"].dropna()
            if len(speedup_cp_numpy) > 0:
                avg_speedup_cp = speedup_cp_numpy.mean()
                summary_table.add_row(
                    algorithm,
                    "[cyan]CPU-Parallel[/cyan]",
                    f"{avg_speedup_cp:.4f}x",
                    "[dim]—[/dim]",
                )

        # GPU row
        gpu_df = alg_df[alg_df["backend"] == "torch"]
        if len(gpu_df) > 0:
            speedup_gpu_numpy = gpu_df["speedup_vs_numpy"].dropna()
            speedup_gpu_cp = gpu_df["speedup_vs_cpu_parallel"].dropna()

            numpy_str = (
                f"{speedup_gpu_numpy.mean():.4f}x"
                if len(speedup_gpu_numpy) > 0
                else "[dim]N/A[/dim]"
            )
            cp_str = (
                f"{speedup_gpu_cp.mean():.4f}x"
                if len(speedup_gpu_cp) > 0
                else "[dim]N/A[/dim]"
            )

            summary_table.add_row(
                algorithm, "[bold red]GPU[/bold red]", numpy_str, cp_str
            )

    console.print("\n")
    console.print(summary_table)


def print_full_results_table(df: pd.DataFrame):
    """Print full results table, sorted by algorithm and then by time (fastest first)."""
    # Sort by algorithm, then by time_seconds (ascending - fastest first)
    sorted_df = df.sort_values(
        by=["algorithm", "time_seconds"], ascending=[True, True]
    ).reset_index(drop=True)

    # Format numeric columns
    display_df = sorted_df.copy()
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

    table = Table(title="[bold]Full Results", box=box.ROUNDED, show_header=True)
    table.add_column("Algorithm", style="cyan")
    table.add_column("Samples", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Permutations", justify="right")
    table.add_column("Backend", style="magenta")
    table.add_column("Time (s)", justify="right", style="green")
    table.add_column("Memory (MB)", justify="right", style="yellow")
    table.add_column("Speedup vs NumPy", justify="right", style="blue")
    table.add_column("Speedup vs CPU-Parallel", justify="right", style="blue")

    for _, row in display_df.iterrows():
        backend_style = {
            "numpy": "[dim]CPU Single[/dim]",
            "cpu-parallel": "[cyan]CPU Parallel[/cyan]",
            "torch": "[bold red]GPU[/bold red]",
        }
        backend_display = backend_style.get(row["backend"], row["backend"])

        table.add_row(
            str(row["algorithm"]),
            str(row["n_samples"]),
            str(row["n_features"]),
            str(row["n_permute"]),
            backend_display,
            str(row["time_seconds"]),
            str(row["memory_mb"]),
            str(row.get("speedup_vs_numpy", "N/A")),
            str(row.get("speedup_vs_cpu_parallel", "N/A")),
        )

    console.print("\n")
    console.print(table)


# ============================================================================
# Interactive Configuration
# ============================================================================


def interactive_config() -> Dict:
    """Interactive configuration using Rich prompts."""
    console.print("\n[bold cyan]Interactive Benchmark Configuration[/bold cyan]")
    console.print("[dim]Press Enter to use defaults[/dim]\n")

    # Algorithms
    valid_algorithms = [
        "one_sample",
        "two_sample",
        "correlation",
        "timeseries_correlation",
        "matrix",
        "isc",
        "isc_group",
    ]
    algo_input = Prompt.ask(
        "Algorithms to benchmark",
        default="one_sample,two_sample,correlation,timeseries_correlation,matrix,isc,isc_group",
        console=console,
    )
    algorithms = [
        x.strip() for x in algo_input.split(",") if x.strip() in valid_algorithms
    ]

    # Sample sizes
    n_samples_input = Prompt.ask(
        "Sample sizes (comma-separated)", default="25,50,100", console=console
    )
    n_samples = [int(x.strip()) for x in n_samples_input.split(",")]

    # Features
    n_features_input = Prompt.ask(
        "Feature/voxel counts (comma-separated)", default="100,1000", console=console
    )
    n_features = [int(x.strip()) for x in n_features_input.split(",")]

    # Permutations
    n_permute_input = Prompt.ask(
        "Permutation counts (comma-separated)", default="1000,5000", console=console
    )
    n_permute = [int(x.strip()) for x in n_permute_input.split(",")]

    # Timepoints
    n_timepoints_input = Prompt.ask(
        "Timepoints for ISC tests (comma-separated)", default="100,500", console=console
    )
    n_timepoints = [int(x.strip()) for x in n_timepoints_input.split(",")]

    # GPU - only ask if GPU is available
    gpu_available, gpu_info = check_gpu_available()
    if gpu_available:
        include_gpu = Confirm.ask(
            "GPU detected include in benchmarking (y)", default=True, console=console
        )
        no_gpu = not include_gpu
    else:
        no_gpu = True
        console.print("[yellow]GPU not available, skipping GPU benchmarks[/yellow]")

    return {
        "algorithms": algorithms,
        "n_samples": n_samples,
        "n_features": n_features,
        "n_permute": n_permute,
        "n_timepoints": n_timepoints,
        "no_gpu": no_gpu,
    }


# ============================================================================
# Benchmark Execution Functions (adapted for Rich progress)
# ============================================================================


def run_one_sample_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    progress: Progress,
    task_id: int,
    quiet: bool = False,
    live_update=None,
) -> int:
    """Run one_sample benchmarks for given device."""
    tests_run = 0

    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        tests_run += 1
        backend_label = (
            "CPU Single"
            if device == "numpy"
            else ("CPU Parallel" if device == "cpu-parallel" else "GPU")
        )
        test_label = f"One Sample Permutation [{backend_label}] | n={n_samples}, f={n_features}, p={n_permute}"

        if not quiet:
            progress.update(task_id, description=test_label)
            if live_update:
                live_update()

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
        progress.update(task_id, advance=1)
        del data
        gc.collect()  # Force garbage collection after each test

    return tests_run


def run_two_sample_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    progress: Progress,
    task_id: int,
    quiet: bool = False,
    live_update=None,
) -> int:
    """Run two_sample benchmarks for given device."""
    tests_run = 0

    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        tests_run += 1
        backend_label = (
            "CPU Single"
            if device == "numpy"
            else ("CPU Parallel" if device == "cpu-parallel" else "GPU")
        )
        test_label = f"Two Sample Permutation [{backend_label}] | n={n_samples}, f={n_features}, p={n_permute}"

        if not quiet:
            progress.update(task_id, description=test_label)
            if live_update:
                live_update()

        data1, data2 = generate_two_sample_data(n_samples, n_samples, n_features)
        condition_key = f"two_sample_{n_samples}_{n_features}_{n_permute}"

        time_result, mem_result = benchmark_two_sample(
            data1,
            data2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

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
        progress.update(task_id, advance=1)
        del data1, data2
        gc.collect()  # Force garbage collection after each test

    return tests_run


def run_correlation_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    progress: Progress,
    task_id: int,
    quiet: bool = False,
    live_update=None,
) -> int:
    """Run correlation benchmarks for given device."""
    tests_run = 0

    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        tests_run += 1
        backend_label = (
            "CPU Single"
            if device == "numpy"
            else ("CPU Parallel" if device == "cpu-parallel" else "GPU")
        )
        test_label = f"Correlation Permutation (pearson) [{backend_label}] | n={n_samples}, f={n_features}, p={n_permute}"

        if not quiet:
            progress.update(task_id, description=test_label)
            if live_update:
                live_update()

        data1, data2 = generate_correlation_data(n_samples, n_features)
        condition_key = f"correlation_{n_samples}_{n_features}_{n_permute}"

        time_result, mem_result = benchmark_correlation(
            data1,
            data2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

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
        progress.update(task_id, advance=1)
        del data1, data2
        gc.collect()  # Force garbage collection after each test

    return tests_run


def run_timeseries_correlation_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    progress: Progress,
    task_id: int,
    quiet: bool = False,
    live_update=None,
) -> int:
    """Run timeseries_correlation benchmarks for given device."""
    tests_run = 0
    methods = ["circle_shift", "phase_randomize"]

    for n_samples, n_features, n_permute in product(
        config["n_samples"], config["n_features"], config["n_permute"]
    ):
        if n_features != 1:
            continue

        for method in methods:
            tests_run += 1
            backend_label = "CPU Parallel" if device == "cpu-parallel" else "GPU"
            method_label = method.replace("_", " ").title()
            test_label = f"Timeseries Correlation ({method_label}) [{backend_label}] | n={n_samples}, p={n_permute}"

            if not quiet:
                progress.update(task_id, description=test_label)
                if live_update:
                    live_update()

            rng = np.random.RandomState(42)
            t = np.linspace(0, 10 * np.pi, n_samples)
            data1 = np.sin(t) + rng.randn(n_samples) * 0.1
            data2 = np.cos(t) + rng.randn(n_samples) * 0.1
            data1 = data1.astype(np.float32)
            data2 = data2.astype(np.float32)

            condition_key = f"timeseries_correlation_{method}_{n_samples}_{n_permute}"

            time_result, mem_result = benchmark_timeseries_correlation(
                data1,
                data2,
                n_permute,
                method,
                parallel=parallel,
                n_jobs=-1 if device == "cpu-parallel" else None,
                random_state=42,
            )

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

            if device == "torch" and condition_key in cpu_parallel_baselines:
                result_dict["speedup_vs_cpu_parallel"] = (
                    cpu_parallel_baselines[condition_key] / time_result
                )
            elif device == "cpu-parallel":
                result_dict["speedup_vs_cpu_parallel"] = 1.0
                cpu_parallel_baselines[condition_key] = time_result

            results.append(result_dict)
            progress.update(task_id, advance=1)
            del data1, data2
            gc.collect()  # Force garbage collection after each test

    return tests_run


def run_matrix_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    progress: Progress,
    task_id: int,
    quiet: bool = False,
    live_update=None,
) -> int:
    """Run matrix benchmarks for given device."""
    tests_run = 0

    if device not in ["numpy", "cpu-parallel"]:
        return 0

    for matrix_size, n_permute in product(config["n_samples"], config["n_permute"]):
        tests_run += 1
        backend_label = "CPU Single" if device == "numpy" else "CPU Parallel"
        test_label = f"Matrix Permutation (Mantel Test, pearson) [{backend_label}] | size={matrix_size}, p={n_permute}"

        if not quiet:
            progress.update(task_id, description=test_label)
            if live_update:
                live_update()

        matrix1, matrix2 = generate_matrix_data(matrix_size)
        condition_key = f"matrix_{matrix_size}_{n_permute}"

        time_result, mem_result = benchmark_matrix(
            matrix1,
            matrix2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
        )

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
        progress.update(task_id, advance=1)
        del matrix1, matrix2
        gc.collect()  # Force garbage collection after each test

    return tests_run


def run_isc_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    progress: Progress,
    task_id: int,
    quiet: bool = False,
    live_update=None,
) -> int:
    """Run isc benchmarks for given device."""
    tests_run = 0
    n_subjects = [10, 20, 30]
    n_timepoints = config.get("n_timepoints", [100, 500])
    n_voxels_config = config["n_features"][:3]

    # Test bootstrap method with both summary statistics
    # Note: circle_shift doesn't support 3D voxel-wise data, so we only test bootstrap
    # For permutation tests with ISC, use bootstrap method (which works with voxel-wise data)
    methods = ["bootstrap"]  # Only bootstrap works with 3D voxel-wise ISC data
    summary_stats = ["leave-one-out", "pairwise"]

    for n_subj, n_tp, n_vox, n_permute, method, summary_stat in product(
        n_subjects,
        n_timepoints,
        n_voxels_config,
        config["n_permute"],
        methods,
        summary_stats,
    ):
        tests_run += 1
        backend_label = "CPU Parallel" if device == "cpu-parallel" else "GPU"
        # Label: "Bootstrap" (only bootstrap method works with voxel-wise ISC data)
        method_label = "Bootstrap"
        summary_label = summary_stat.replace("-", " ").title()
        test_label = (
            f"ISC ({method_label}, {summary_label}) [{backend_label}] | "
            f"subjects={n_subj}, timepoints={n_tp}, voxels={n_vox}, p={n_permute}"
        )

        if not quiet:
            progress.update(task_id, description=test_label)
            if live_update:
                live_update()

        data = generate_isc_data(n_tp, n_subj, n_vox)
        condition_key = (
            f"isc_{method}_{summary_stat}_{n_subj}_{n_tp}_{n_vox}_{n_permute}"
        )

        time_result, mem_result = benchmark_isc(
            data,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
            method=method,
            summary_statistic=summary_stat,
        )

        # Algorithm name: "isc_bootstrap_loo" or "isc_bootstrap_pairwise"
        # Note: Only bootstrap method works with voxel-wise ISC data
        algorithm_name = f"isc_{method}_{summary_stat}"
        result_dict = {
            "algorithm": algorithm_name,
            "n_samples": n_tp,
            "n_features": n_vox,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

        if device == "torch" and condition_key in cpu_parallel_baselines:
            result_dict["speedup_vs_cpu_parallel"] = (
                cpu_parallel_baselines[condition_key] / time_result
            )
        elif device == "cpu-parallel":
            result_dict["speedup_vs_cpu_parallel"] = 1.0
            cpu_parallel_baselines[condition_key] = time_result

        results.append(result_dict)
        progress.update(task_id, advance=1)
        del data
        gc.collect()  # Force garbage collection after each test

    return tests_run


def run_isc_group_benchmarks(
    config: Dict,
    device: str,
    parallel: Optional[str],
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List[Dict],
    progress: Progress,
    task_id: int,
    quiet: bool = False,
    live_update=None,
) -> int:
    """Run isc_group benchmarks for given device."""
    tests_run = 0
    group_sizes = [(10, 10), (20, 20), (30, 30)]
    n_timepoints = config.get("n_timepoints", [100, 500])
    n_voxels_config = config["n_features"][:3]

    # Test both methods and summary statistics
    methods = ["permute", "bootstrap"]
    summary_stats = ["pairwise", "leave-one-out"]

    if device not in ["numpy", "cpu-parallel"]:
        return 0

    for (n_subj1, n_subj2), n_tp, n_vox, n_permute, method, summary_stat in product(
        group_sizes,
        n_timepoints,
        n_voxels_config,
        config["n_permute"],
        methods,
        summary_stats,
    ):
        tests_run += 1
        backend_label = "CPU Single" if device == "numpy" else "CPU Parallel"
        # Use clearer labels: "Bootstrap" or "Permutation"
        if method == "permute":
            method_label = "Permutation"
        else:
            method_label = "Bootstrap"
        summary_label = summary_stat.replace("-", " ").title()
        test_label = (
            f"ISC Group ({method_label}, {summary_label}) [{backend_label}] | "
            f"group1={n_subj1}, group2={n_subj2}, timepoints={n_tp}, voxels={n_vox}, p={n_permute}"
        )

        if not quiet:
            progress.update(task_id, description=test_label)
            if live_update:
                live_update()

        group1 = generate_isc_data(n_tp, n_subj1, n_vox)
        group2 = generate_isc_data(n_tp, n_subj2, n_vox)
        condition_key = f"isc_group_{method}_{summary_stat}_{n_subj1}_{n_subj2}_{n_tp}_{n_vox}_{n_permute}"

        time_result, mem_result = benchmark_isc_group(
            group1,
            group2,
            n_permute,
            parallel=parallel,
            n_jobs=-1 if device == "cpu-parallel" else None,
            random_state=42,
            method=method,
            summary_statistic=summary_stat,
        )

        # Use clearer algorithm name: "isc_group_bootstrap_loo" or "isc_group_permutation_pairwise"
        algorithm_name = f"isc_group_{method}_{summary_stat}"
        result_dict = {
            "algorithm": algorithm_name,
            "n_samples": f"{n_tp},{n_subj1},{n_subj2}",
            "n_features": n_vox,
            "n_permute": n_permute,
            "backend": device,
            "time_seconds": time_result,
            "memory_mb": mem_result,
            "speedup_vs_numpy": None,
            "speedup_vs_cpu_parallel": None,
        }

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
        progress.update(task_id, advance=1)
        del group1, group2
        gc.collect()  # Force garbage collection after each test

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
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "two_sample":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "correlation":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "timeseries_correlation":
            for n_samples, n_features, n_permute in product(
                config["n_samples"], config["n_features"], config["n_permute"]
            ):
                if n_features == 1:
                    device_counts["cpu-parallel"] += 2
                    if gpu_available and not no_gpu:
                        device_counts["torch"] += 2
        elif algorithm == "matrix":
            for matrix_size, n_permute in product(
                config["n_samples"], config["n_permute"]
            ):
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1
        elif algorithm == "isc":
            n_subjects = [10, 20, 30]
            n_timepoints = config.get("n_timepoints", [100, 500])
            n_voxels = config["n_features"][:3]
            methods = ["bootstrap"]  # Only bootstrap works with 3D voxel-wise ISC data
            summary_stats = ["leave-one-out", "pairwise"]
            for n_subj, n_tp, n_vox, n_permute, method, summary_stat in product(
                n_subjects,
                n_timepoints,
                n_voxels,
                config["n_permute"],
                methods,
                summary_stats,
            ):
                device_counts["cpu-parallel"] += 1
                if gpu_available and not no_gpu:
                    device_counts["torch"] += 1
        elif algorithm == "isc_group":
            group_sizes = [(10, 10), (20, 20), (30, 30)]
            n_timepoints = config.get("n_timepoints", [100, 500])
            n_voxels = config["n_features"][:3]
            methods = ["permute", "bootstrap"]
            summary_stats = ["pairwise", "leave-one-out"]
            for (n1, n2), n_tp, n_vox, n_permute, method, summary_stat in product(
                group_sizes,
                n_timepoints,
                n_voxels,
                config["n_permute"],
                methods,
                summary_stats,
            ):
                device_counts["numpy"] += 1
                device_counts["cpu-parallel"] += 1

    total = sum(device_counts.values())
    return {"device_counts": device_counts, "total": total}


def run_systematic_benchmarks(config: Dict) -> pd.DataFrame:
    """Run systematic benchmark grid with Rich Live layout."""
    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()

    # Count total tests
    test_counts = count_total_tests(config, gpu_available, config.get("no_gpu", False))
    device_counts = test_counts["device_counts"]

    # Results storage
    results = []
    numpy_baselines = {}
    cpu_parallel_baselines = {}
    all_device_results = []  # Accumulate all results

    # Determine which devices to run
    devices_to_run = []
    if device_counts["numpy"] > 0:
        devices_to_run.append(("numpy", None))
    if device_counts["cpu-parallel"] > 0:
        devices_to_run.append(("cpu-parallel", "cpu"))
    if device_counts["torch"] > 0:
        devices_to_run.append(("torch", "gpu"))

    if config.get("quiet", False):
        # Quiet mode: just run without display
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
            disable=True,
        ) as progress:
            return _run_benchmarks_internal(
                config,
                devices_to_run,
                device_counts,
                numpy_baselines,
                cpu_parallel_baselines,
                results,
                progress,
                None,
            )

    # Create layout for Live display
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),  # Device name (larger for visibility)
        Layout(name="status", size=2),  # Running test name + system info line
        Layout(name="progress", size=3),  # Progress bar
        Layout(name="results", size=None),  # Results table (grows - expanded space)
    )

    # Initialize layout components
    current_device = ""
    current_test = "[dim]Waiting to start...[/dim]"
    system_info_line = format_system_info_line(config, gpu_available, gpu_info)
    results_table = create_results_table([])

    # Create progress bar that will be updated
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    )

    def make_layout() -> Layout:
        """Build the complete layout."""
        layout["header"].update(
            Panel(
                Text(f"Device: {current_device}", style="bold yellow"),
                border_style="yellow",
                padding=(0, 1),
            )
        )
        # Combine test name and system info in status section
        status_text = f"{current_test}\n[dim]{system_info_line}[/dim]"
        layout["status"].update(
            Panel(Text(status_text, style="cyan"), border_style="cyan", padding=(0, 1))
        )
        layout["progress"].update(progress)
        layout["results"].update(
            Panel(results_table, title="[bold green]Test Results", border_style="green")
        )
        return layout

    # Run benchmarks with Live display
    with Live(make_layout(), refresh_per_second=10, screen=True) as live_context:

        def live_update():
            """Update the live display."""
            live_context.update(make_layout())

        for device_idx, (device, parallel) in enumerate(devices_to_run, 1):
            if device == "cpu-parallel":
                n_jobs = os.cpu_count() or 1
            else:
                n_jobs = None

            device_label = get_device_label(device, n_jobs)
            current_device = device_label
            current_test = "[dim]Initializing...[/dim]"
            live_context.update(make_layout())

            device_results = []
            device_task_id = progress.add_task(
                "",
                total=device_counts[device],
            )

            # Remove the initial status update - it will be updated by progress
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
                        progress,
                        device_task_id,
                        False,
                        live_update,
                    )
                    device_results.extend(results[-tests_run:])
                    # Update display after each algorithm
                    all_device_results.extend(results[-tests_run:])
                    results_table = create_results_table(all_device_results)
                    if progress.tasks and device_task_id < len(progress.tasks):
                        task = progress.tasks[device_task_id]
                        if task.description:
                            current_test = task.description
                    live_context.update(make_layout())
                elif algorithm == "two_sample":
                    tests_run = run_two_sample_benchmarks(
                        config,
                        device,
                        parallel,
                        numpy_baselines,
                        cpu_parallel_baselines,
                        results,
                        progress,
                        device_task_id,
                        False,
                        live_update,
                    )
                    device_results.extend(results[-tests_run:])
                    # Update display after each algorithm
                    all_device_results.extend(results[-tests_run:])
                    results_table = create_results_table(all_device_results)
                    if progress.tasks and device_task_id < len(progress.tasks):
                        task = progress.tasks[device_task_id]
                        if task.description:
                            current_test = task.description
                    live_context.update(make_layout())
                elif algorithm == "correlation":
                    tests_run = run_correlation_benchmarks(
                        config,
                        device,
                        parallel,
                        numpy_baselines,
                        cpu_parallel_baselines,
                        results,
                        progress,
                        device_task_id,
                        False,
                        live_update,
                    )
                    device_results.extend(results[-tests_run:])
                    # Update display after each algorithm
                    all_device_results.extend(results[-tests_run:])
                    results_table = create_results_table(all_device_results)
                    if progress.tasks and device_task_id < len(progress.tasks):
                        task = progress.tasks[device_task_id]
                        if task.description:
                            current_test = task.description
                    live_context.update(make_layout())
                elif algorithm == "timeseries_correlation":
                    tests_run = run_timeseries_correlation_benchmarks(
                        config,
                        device,
                        parallel,
                        cpu_parallel_baselines,
                        results,
                        progress,
                        device_task_id,
                        False,
                        live_update,
                    )
                    device_results.extend(results[-tests_run:])
                    # Update display after each algorithm
                    all_device_results.extend(results[-tests_run:])
                    results_table = create_results_table(all_device_results)
                    if progress.tasks and device_task_id < len(progress.tasks):
                        task = progress.tasks[device_task_id]
                        if task.description:
                            current_test = task.description
                    live_context.update(make_layout())
                elif algorithm == "matrix":
                    tests_run = run_matrix_benchmarks(
                        config,
                        device,
                        parallel,
                        numpy_baselines,
                        cpu_parallel_baselines,
                        results,
                        progress,
                        device_task_id,
                        False,
                        live_update,
                    )
                    device_results.extend(results[-tests_run:])
                    # Update display after each algorithm
                    all_device_results.extend(results[-tests_run:])
                    results_table = create_results_table(all_device_results)
                    if progress.tasks and device_task_id < len(progress.tasks):
                        task = progress.tasks[device_task_id]
                        if task.description:
                            current_test = task.description
                    live_context.update(make_layout())
                elif algorithm == "isc":
                    tests_run = run_isc_benchmarks(
                        config,
                        device,
                        parallel,
                        cpu_parallel_baselines,
                        results,
                        progress,
                        device_task_id,
                        False,
                        live_update,
                    )
                    device_results.extend(results[-tests_run:])
                    # Update display after each algorithm
                    all_device_results.extend(results[-tests_run:])
                    results_table = create_results_table(all_device_results)
                    if progress.tasks and device_task_id < len(progress.tasks):
                        task = progress.tasks[device_task_id]
                        if task.description:
                            current_test = task.description
                    live_context.update(make_layout())
                elif algorithm == "isc_group":
                    tests_run = run_isc_group_benchmarks(
                        config,
                        device,
                        parallel,
                        numpy_baselines,
                        cpu_parallel_baselines,
                        results,
                        progress,
                        device_task_id,
                        False,
                        live_update,
                    )
                    device_results.extend(results[-tests_run:])
                    # Update display after each algorithm
                    all_device_results.extend(results[-tests_run:])
                    results_table = create_results_table(all_device_results)
                    if progress.tasks and device_task_id < len(progress.tasks):
                        task = progress.tasks[device_task_id]
                        if task.description:
                            current_test = task.description
                    live_context.update(make_layout())

            # Accumulate results at end of device
            all_device_results.extend(device_results)
            results_table = create_results_table(all_device_results)
            current_test = f"[green]✓ Completed {device_label}[/green]"
            live_context.update(make_layout())

            # Force garbage collection after each device completes
            del device_results
            gc.collect()

            # Check memory usage and warn if high
            try:
                mem = psutil.virtual_memory()
                if mem.percent > 85:
                    console.print(
                        f"[yellow]Warning: Memory usage is {mem.percent:.1f}%. "
                        f"Consider reducing problem sizes or running fewer algorithms.[/yellow]",
                        style="dim",
                    )
            except Exception:
                pass  # Ignore memory check errors

    console.print("\n[bold green]✓ Benchmark Complete![/bold green]\n")
    return pd.DataFrame(results)


def _run_benchmarks_internal(
    config: Dict,
    devices_to_run: List,
    device_counts: Dict,
    numpy_baselines: Dict,
    cpu_parallel_baselines: Dict,
    results: List,
    progress: Progress,
    update_callback,
) -> pd.DataFrame:
    """Internal benchmark execution logic."""
    for device_idx, (device, parallel) in enumerate(devices_to_run, 1):
        device_task_id = progress.add_task("", total=device_counts[device])

        for algorithm in config["algorithms"]:
            if algorithm == "one_sample":
                run_one_sample_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    progress,
                    device_task_id,
                    True,
                )
            elif algorithm == "two_sample":
                run_two_sample_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    progress,
                    device_task_id,
                    True,
                )
            elif algorithm == "correlation":
                run_correlation_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    progress,
                    device_task_id,
                    True,
                )
            elif algorithm == "timeseries_correlation":
                run_timeseries_correlation_benchmarks(
                    config,
                    device,
                    parallel,
                    cpu_parallel_baselines,
                    results,
                    progress,
                    device_task_id,
                    True,
                )
            elif algorithm == "matrix":
                run_matrix_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    progress,
                    device_task_id,
                    True,
                )
            elif algorithm == "isc":
                run_isc_benchmarks(
                    config,
                    device,
                    parallel,
                    cpu_parallel_baselines,
                    results,
                    progress,
                    device_task_id,
                    True,
                )
            elif algorithm == "isc_group":
                run_isc_group_benchmarks(
                    config,
                    device,
                    parallel,
                    numpy_baselines,
                    cpu_parallel_baselines,
                    results,
                    progress,
                    device_task_id,
                    True,
                )

    return pd.DataFrame(results)


# ============================================================================
# CLI and Configuration
# ============================================================================


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Systematic inference algorithm benchmarks for neuroimaging (Rich-enhanced)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (prompts for configuration)
  %(prog)s

  # CLI mode with defaults
  %(prog)s --algorithm one_sample --n-features "1,100"

  # Skip GPU
  %(prog)s --no-gpu
        """,
    )

    parser.add_argument(
        "-a",
        "--algorithm",
        type=str,
        default=None,
        help="Algorithms to benchmark (comma-separated). If not provided, will prompt interactively.",
    )

    parser.add_argument(
        "-n",
        "--n-samples",
        type=str,
        default=None,
        help="Sample sizes (comma-separated)",
    )

    parser.add_argument(
        "-f",
        "--n-features",
        type=str,
        default=None,
        help="Feature/voxel counts (comma-separated)",
    )

    parser.add_argument(
        "-t",
        "--n-timepoints",
        type=str,
        default=None,
        help="Timepoints for ISC tests (comma-separated)",
    )

    parser.add_argument(
        "-p",
        "--n-permute",
        type=str,
        default=None,
        help="Permutation counts (comma-separated)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show benchmark plan without running",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Skip GPU benchmarks",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results_inference_systematic.csv",
        help="Output CSV filename (only used with --save flag, default: results_inference_systematic.csv)",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        help="Save results to CSV file",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    return parser.parse_args()


def main():
    """Run systematic benchmarks with CLI support and save results."""
    args = parse_args()

    # Interactive mode if no CLI args provided
    if not any(
        [
            args.algorithm,
            args.n_samples,
            args.n_features,
            args.n_timepoints,
            args.n_permute,
            args.dry_run,
            args.no_gpu,
        ]
    ):
        config = interactive_config()
    else:
        # Parse CLI arguments
        try:
            algorithms = (
                [x.strip() for x in args.algorithm.split(",")]
                if args.algorithm
                else [
                    "one_sample",
                    "two_sample",
                    "correlation",
                    "timeseries_correlation",
                    "matrix",
                    "isc",
                    "isc_group",
                ]
            )
            valid_algorithms = [
                "one_sample",
                "two_sample",
                "correlation",
                "timeseries_correlation",
                "matrix",
                "isc",
                "isc_group",
            ]
            algorithms = [a for a in algorithms if a in valid_algorithms]
            if not algorithms:
                console.print("[red]Error: No valid algorithms specified[/red]")
                return

            n_samples = (
                [int(x.strip()) for x in args.n_samples.split(",")]
                if args.n_samples
                else [25]
            )
            n_features = (
                [int(x.strip()) for x in args.n_features.split(",")]
                if args.n_features
                else [100]
            )
            n_permute = (
                [int(x.strip()) for x in args.n_permute.split(",")]
                if args.n_permute
                else [5000]
            )
            n_timepoints = (
                [int(x.strip()) for x in args.n_timepoints.split(",")]
                if args.n_timepoints
                else [100, 500]
            )
        except ValueError as e:
            console.print(f"[red]Error parsing arguments: {e}[/red]")
            return

        config = {
            "algorithms": algorithms,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_permute": n_permute,
            "n_timepoints": n_timepoints,
            "no_gpu": args.no_gpu,
            "quiet": args.quiet,
        }

    # Check GPU availability
    gpu_available, gpu_info = check_gpu_available()

    config["gpu_available"] = gpu_available
    config["gpu_device"] = gpu_info["device"] if gpu_available else "none"

    # Handle dry-run
    if args.dry_run:
        console.print("\n[bold yellow]DRY RUN: Benchmark Plan[/bold yellow]\n")
        print_system_info(config, gpu_available, gpu_info)
        print_benchmark_config(config)

        test_counts = count_total_tests(
            config, gpu_available, config.get("no_gpu", False)
        )
        summary_table = Table.grid(padding=1)
        summary_table.add_column(style="cyan", justify="right")
        summary_table.add_column(style="white")
        summary_table.add_row(
            "CPU (NumPy):", str(test_counts["device_counts"]["numpy"])
        )
        summary_table.add_row(
            "CPU-Parallel:", str(test_counts["device_counts"]["cpu-parallel"])
        )
        if gpu_available and not config.get("no_gpu", False):
            summary_table.add_row(
                "GPU (PyTorch):", str(test_counts["device_counts"]["torch"])
            )
        summary_table.add_row("Total:", f"[bold]{test_counts['total']}[/bold]")
        console.print(
            Panel(
                summary_table, title="[bold yellow]Test Counts", border_style="yellow"
            )
        )
        console.print("\n[dim]To run: Remove --dry-run flag[/dim]\n")
        return

    # Run benchmarks
    results_df = run_systematic_benchmarks(config)

    # Save to CSV with timestamp in results subdirectory (only if --save flag provided)
    if args.save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(results_dir, exist_ok=True)

        csv_base = args.output.replace(".csv", "")
        csv_filename = f"{csv_base}_{timestamp}.csv"
        csv_path = os.path.join(results_dir, csv_filename)
        results_df.to_csv(csv_path, index=False)

        if not config.get("quiet", False):
            console.print(f"[green]Results saved to:[/green] {csv_path}\n")
    else:
        if not config.get("quiet", False):
            console.print(
                "[dim]Results not saved (use --save flag to save CSV file)[/dim]\n"
            )

    if not config.get("quiet", False):
        # Print summary tables
        print_results_summary_table(results_df)

        # Always print full results table (sorted by fastest device for each algorithm)
        print_full_results_table(results_df)


if __name__ == "__main__":
    main()
