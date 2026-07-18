"""Run nltools benchmarks and write a results artifact + provenance sidecar.

    uv run python -m benchmarks.run                 # all domains, full sizes
    uv run python -m benchmarks.run --quick         # tiny sizes (validation)
    uv run python -m benchmarks.run --domains ridge collection
    uv run python -m benchmarks.run --dry-run       # list what would run

Writes ``benchmarks/results/<host>[-<tag>].parquet`` + ``<...>.env.json``. Docs
(`docs/performance.md`) are regenerated from that parquet, not hand-edited.
"""

from __future__ import annotations

import argparse

from benchmarks import bench_collection, bench_inference, bench_predict, bench_ridge
from benchmarks.harness import BenchResult, write_results

DOMAINS = {
    "ridge": bench_ridge,
    "inference": bench_inference,
    "predict": bench_predict,
    "collection": bench_collection,
}


def _print_table(results: list[BenchResult]) -> None:
    header = f"{'domain':11s} {'condition':44s} {'device':6s} {'sec':>9s} {'rss_MB':>9s} {'gpu_MB':>8s}"
    print(header)
    print("-" * len(header))
    for r in results:
        gpu = f"{r.peak_device_mb:.0f}" if r.peak_device_mb is not None else "-"
        print(
            f"{r.domain:11s} {r.name:44s} {r.device:6s} "
            f"{r.seconds:9.4f} {r.peak_rss_mb:9.1f} {gpu:>8s}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--domains",
        nargs="*",
        default=list(DOMAINS),
        choices=list(DOMAINS),
        help="Which benchmark domains to run (default: all).",
    )
    ap.add_argument("--reps", type=int, default=3, help="Timed reps per condition.")
    ap.add_argument("--quick", action="store_true", help="Tiny sizes for validation.")
    ap.add_argument(
        "--device",
        default="cpu",
        help="Provenance label for the run's primary device (cpu/mps/cuda).",
    )
    ap.add_argument("--tag", default=None, help="Optional suffix on the artifact name.")
    ap.add_argument("--out", default="benchmarks/results", help="Output directory.")
    ap.add_argument(
        "--dry-run", action="store_true", help="List domains and exit without running."
    )
    args = ap.parse_args()

    if args.dry_run:
        print(f"Would run domains: {', '.join(args.domains)}")
        print(f"  reps={args.reps} quick={args.quick} -> {args.out}")
        return

    all_results: list[BenchResult] = []
    for domain in args.domains:
        print(f"\n=== {domain} ===")
        results = DOMAINS[domain].run(reps=args.reps, quick=args.quick)
        all_results.extend(results)
        _print_table(results)

    print("\n=== summary ===")
    _print_table(all_results)

    parquet_path, env_path = write_results(
        all_results, args.out, device=args.device, tag=args.tag
    )
    print(f"\nwrote {parquet_path}\nwrote {env_path}")


if __name__ == "__main__":
    main()
