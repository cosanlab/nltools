"""Regenerate the numeric tables in docs/performance.md from a results artifact.

Reads every ``benchmarks/results/*.parquet`` (+ its ``.env.json`` provenance)
and splices Markdown tables between the ``<!-- BENCH:START -->`` /
``<!-- BENCH:END -->`` markers in ``docs/performance.md``. The surrounding prose
(guide, recommendations) is never touched — only the generated block.

    uv run python -m benchmarks.build_docs

Tables produced:
- **Speed + memory per domain** — every condition (sec, peak RSS, GPU MB).
- **GPU speedup** (ridge, inference) — CPU vs GPU seconds paired by condition.
- **Memory: lazy vs in-memory** (collection) — peak RSS as N subjects grows.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "benchmarks" / "results"
PERF_DOC = REPO / "docs" / "performance.md"
START = "<!-- BENCH:START -->"
END = "<!-- BENCH:END -->"


def _load() -> tuple[pl.DataFrame, dict]:
    frames = [pl.read_parquet(p) for p in sorted(RESULTS_DIR.glob("*.parquet"))]
    if not frames:
        raise SystemExit(f"no parquet artifacts in {RESULTS_DIR}")
    df = pl.concat(frames, how="diagonal_relaxed")
    env_files = sorted(RESULTS_DIR.glob("*.env.json"))
    env = json.loads(env_files[0].read_text()) if env_files else {}
    return df, env


def _fmt_sec(x: float | None) -> str:
    if x is None:
        return "-"
    return f"{x * 1000:.1f} ms" if x < 1 else f"{x:.2f} s"


def _provenance(env: dict) -> str:
    bits = [
        f"**Host:** {env.get('host', '?')}",
        f"**Platform:** {env.get('platform', '?')}",
        f"**Python:** {env.get('python', '?')}",
        f"**NumPy:** {env.get('numpy', '?')}",
        f"**PyTorch:** {env.get('torch', 'n/a')}",
        f"**nltools:** {env.get('nltools', '?')}",
    ]
    mps = env.get("mps_available")
    cuda = env.get("cuda_available")
    dev = "MPS" if mps else ("CUDA" if cuda else "CPU only")
    bits.append(f"**GPU:** {dev}")
    return "  \n".join(bits)


def _domain_table(df: pl.DataFrame, domain: str) -> str:
    sub = df.filter(pl.col("domain") == domain).sort("name", "device")
    lines = [
        f"#### {domain}",
        "",
        "| Condition | Device | Time | Peak RSS | GPU mem |",
        "|---|---|--:|--:|--:|",
    ]
    for r in sub.iter_rows(named=True):
        gpu = f"{r['peak_device_mb']:.0f} MB" if r.get("peak_device_mb") else "-"
        lines.append(
            f"| {r['name']} | {r['device']} | {_fmt_sec(r['seconds'])} "
            f"| {r['peak_rss_mb']:.1f} MB | {gpu} |"
        )
    return "\n".join(lines)


def _speedup_table(df: pl.DataFrame, domain: str) -> str:
    """CPU vs GPU seconds paired by condition name."""
    sub = df.filter(pl.col("domain") == domain)
    piv = (
        sub.group_by("name", "device")
        .agg(pl.col("seconds").median())
        .pivot(values="seconds", index="name", on="device")
    )
    cols = piv.columns
    if "cpu" not in cols or not ({"mps", "cuda"} & set(cols)):
        return ""
    gpu_col = "mps" if "mps" in cols else "cuda"
    piv = piv.with_columns((pl.col("cpu") / pl.col(gpu_col)).alias("speedup")).sort(
        "name"
    )
    lines = [
        f"#### {domain} — GPU speedup ({gpu_col})",
        "",
        f"| Condition | CPU | {gpu_col.upper()} | Speedup |",
        "|---|--:|--:|--:|",
    ]
    for r in piv.iter_rows(named=True):
        if r["cpu"] is None or r[gpu_col] is None:
            continue
        lines.append(
            f"| {r['name']} | {_fmt_sec(r['cpu'])} | {_fmt_sec(r[gpu_col])} "
            f"| **{r['speedup']:.2f}×** |"
        )
    return "\n".join(lines)


def _collection_memory_table(df: pl.DataFrame) -> str:
    sub = df.filter((pl.col("domain") == "collection") & (pl.col("p_op") == "mean"))
    if sub.is_empty():
        return ""
    piv = (
        sub.group_by("p_n_subjects", "p_mode")
        .agg(pl.col("peak_rss_mb").max())
        .pivot(values="peak_rss_mb", index="p_n_subjects", on="p_mode")
        .sort("p_n_subjects")
    )
    lines = [
        "#### collection — peak RSS: lazy vs in-memory `.mean()`",
        "",
        "| N subjects | lazy | in-memory |",
        "|---:|--:|--:|",
    ]
    for r in piv.iter_rows(named=True):
        lazy = f"{r['lazy']:.1f} MB" if r.get("lazy") is not None else "-"
        inmem = f"{r['in_memory']:.1f} MB" if r.get("in_memory") is not None else "-"
        lines.append(f"| {r['p_n_subjects']} | {lazy} | {inmem} |")
    return "\n".join(lines)


def build_block() -> str:
    df, env = _load()
    parts = [
        ":::{note} Auto-generated",
        "This block is generated from `benchmarks/results/*.parquet` by "
        "`uv run python -m benchmarks.build_docs`. Do not edit by hand.",
        ":::",
        "",
        _provenance(env),
        "",
    ]
    speedups = [t for d in ("ridge", "inference") if (t := _speedup_table(df, d))]
    if speedups:
        parts += ["### GPU speedup", "", *_join(speedups)]
    mem = _collection_memory_table(df)
    if mem:
        parts += ["### Memory scaling", "", mem, ""]
    parts += ["### Full results", ""]
    for domain in ("ridge", "predict", "inference", "collection"):
        if not df.filter(pl.col("domain") == domain).is_empty():
            parts += [_domain_table(df, domain), ""]
    return "\n".join(parts).rstrip() + "\n"


def _join(tables: list[str]) -> list[str]:
    out: list[str] = []
    for t in tables:
        out += [t, ""]
    return out


def splice() -> None:
    block = build_block()
    text = PERF_DOC.read_text()
    if START not in text or END not in text:
        raise SystemExit(
            f"markers {START}/{END} not found in {PERF_DOC}; add them first."
        )
    pre, rest = text.split(START, 1)
    _, post = rest.split(END, 1)
    PERF_DOC.write_text(f"{pre}{START}\n{block}{END}{post}")
    print(f"regenerated benchmark block in {PERF_DOC}")


if __name__ == "__main__":
    splice()
