#!/usr/bin/env python
"""Download and stage browser-sized datasets for the tutorials.

Covers the GLM, MVPA, encoding, and ISC tutorials. The outputs are written below
``scratch/tutorial-data/tutorials`` by default. That directory is intentionally
gitignored: this script is the reproducible source of the staged data, while the
generated files are intended for upload to the ``nltools/niftis`` Hugging Face
dataset (under ``tutorials/{glm,mvpa,encoding,isc}/``).

The encoding tutorial (nilearn Miyawaki2008) and ISC tutorial (nilearn
development_fmri) both need their full per-run/per-subject timecourses — encoding
pairs every volume with a stimulus, ISC correlates full timecourses across
subjects — so neither is temporally trimmed. Their 4D BOLD is still quantized to
scaled int16 to cut transfer size, mirroring the GLM trim. The encoding stage
keeps the first ``--miyawaki-runs`` runs (func + 10x10 stimulus labels + the
subject-native mask); the ISC stage keeps ``--isc-subjects`` subjects' func.

The GLM tutorial performs an eight-subject group analysis, so its default trim
keeps subjects 01--08 and the first 76 volumes of each run.  Those volumes
contain four complete ``language`` and four complete ``string`` blocks.  The
matching events and confounds rows are trimmed to the same run duration.  The
float32 BOLD files are quantized to scaled int16 NIfTI files to reduce transfer
size without changing their affine, header geometry, or values as loaded by
nibabel/nltools beyond the recorded quantization tolerance.

The MVPA tutorial needs every Haxby object category, face/house decoding, and
the original rest volumes for its two-TR RSA shift.  Every Haxby session has
all of those, so the default keeps one complete session and drops the other 11
sessions while preserving the full spatial extent and resolution.

Usage::

    python scripts/make_tutorial_data.py
    python scripts/make_tutorial_data.py --dataset glm
    python scripts/make_tutorial_data.py --dataset mvpa --haxby-session 0
    python scripts/make_tutorial_data.py --output-dir scratch/tutorial-data/tutorials
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from nibabel.spatialimages import SpatialImage
from nilearn.datasets import (
    fetch_development_fmri,
    fetch_haxby,
    fetch_language_localizer_demo_dataset,
    fetch_miyawaki2008,
)
from nilearn.interfaces.bids import get_bids_files

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "scratch" / "tutorial-data" / "tutorials"
GLM_SUBJECTS = tuple(f"{subject:02d}" for subject in range(1, 9))


def byte_size(path: Path) -> int:
    """Return the recursive byte size of a file or directory."""
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def copy_file(source: Path, destination: Path) -> Path:
    """Copy one file, creating its destination directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def save_like(source: SpatialImage, data: np.ndarray, destination: Path) -> Path:
    """Save ``data`` with the source image's affine and spatial metadata."""
    header = source.header.copy()
    image = nib.Nifti1Image(data, source.affine, header=header)
    qform, qform_code = source.get_qform(coded=True)
    sform, sform_code = source.get_sform(coded=True)
    image.set_qform(qform, int(qform_code))
    image.set_sform(sform, int(sform_code))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, destination)
    return destination


def save_scaled_int16(
    source_path: Path, volumes: int, destination: Path
) -> tuple[Path, float]:
    """Trim a 4D image and save scaled int16 data; return max load-time error."""
    source = nib.load(source_path)
    data = np.asarray(source.dataobj[..., :volumes], dtype=np.float32)
    max_abs = float(np.nanmax(np.abs(data)))
    slope = max_abs / np.iinfo(np.int16).max if max_abs else 1.0
    limits = np.iinfo(np.int16)
    quantized = np.rint(data / slope).clip(limits.min, limits.max).astype(np.int16)

    header = source.header.copy()
    header.set_data_dtype(np.int16)
    image = nib.Nifti1Image(quantized, source.affine, header=header)
    image.header.set_slope_inter(slope, 0.0)
    qform, qform_code = source.get_qform(coded=True)
    sform, sform_code = source.get_sform(coded=True)
    image.set_qform(qform, int(qform_code))
    image.set_sform(sform, int(sform_code))
    destination.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, destination)

    loaded = nib.load(destination)
    restored = np.asarray(loaded.dataobj, dtype=np.float32)
    max_error = float(np.nanmax(np.abs(restored - data)))
    return destination, max_error


def glm_subject_files(data_dir: Path, subject: str) -> dict[str, Path]:
    """Resolve exactly the four BIDS files read by the GLM tutorial."""
    derivatives = data_dir / "derivatives"
    return {
        "bold": Path(
            get_bids_files(
                derivatives, file_tag="bold", file_type="nii.gz", sub_label=subject
            )[0]
        ),
        "events": Path(
            get_bids_files(
                data_dir, file_tag="events", file_type="tsv", sub_label=subject
            )[0]
        ),
        "confounds": Path(
            get_bids_files(
                derivatives, file_type="tsv", modality_folder="func", sub_label=subject
            )[0]
        ),
        "sidecar": Path(
            get_bids_files(
                derivatives, file_tag="bold", file_type="json", sub_label=subject
            )[0]
        ),
    }


def stage_glm(output_root: Path, volumes: int) -> dict[str, Any]:
    """Stage the eight-subject, temporally trimmed language-localizer dataset."""
    dataset = fetch_language_localizer_demo_dataset(verbose=1)
    source_root = Path(dataset["data_dir"])
    destination_root = output_root / "glm"
    if destination_root.exists():
        shutil.rmtree(destination_root)

    source_files: list[Path] = []
    staged_files: list[Path] = []
    quantization_errors: dict[str, float] = {}

    for metadata in (
        "dataset_description.json",
        "derivatives/dataset_description.json",
    ):
        source = source_root / metadata
        source_files.append(source)
        staged_files.append(copy_file(source, destination_root / metadata))

    for subject in GLM_SUBJECTS:
        files = glm_subject_files(source_root, subject)
        source_files.extend(files.values())
        bold_image = nib.load(files["bold"])
        if volumes > bold_image.shape[3]:
            raise ValueError(
                f"--glm-volumes={volumes} exceeds sub-{subject}'s {bold_image.shape[3]} volumes"
            )
        tr = float(json.loads(files["sidecar"].read_text())["RepetitionTime"])
        run_duration = volumes * tr

        bold_relative = files["bold"].relative_to(source_root)
        bold_destination = destination_root / bold_relative
        _, max_error = save_scaled_int16(files["bold"], volumes, bold_destination)
        staged_files.append(bold_destination)
        quantization_errors[subject] = max_error

        events = pd.read_csv(files["events"], sep="\t")
        events = events.loc[events["onset"] + events["duration"] <= run_duration].copy()
        if set(events["trial_type"]) != {"language", "string"}:
            raise ValueError(
                f"sub-{subject}: trimmed events do not contain both conditions"
            )
        events_destination = destination_root / files["events"].relative_to(source_root)
        events_destination.parent.mkdir(parents=True, exist_ok=True)
        events.to_csv(events_destination, sep="\t", index=False)
        staged_files.append(events_destination)

        confounds = pd.read_csv(files["confounds"], sep="\t").iloc[:volumes].copy()
        if len(confounds) != volumes:
            raise ValueError(f"sub-{subject}: confounds has fewer than {volumes} rows")
        confounds_destination = destination_root / files["confounds"].relative_to(
            source_root
        )
        confounds_destination.parent.mkdir(parents=True, exist_ok=True)
        confounds.to_csv(confounds_destination, sep="\t", index=False)
        staged_files.append(confounds_destination)

        sidecar_destination = destination_root / files["sidecar"].relative_to(
            source_root
        )
        staged_files.append(copy_file(files["sidecar"], sidecar_destination))

    return {
        "source_root": source_root,
        "source_dataset_bytes": byte_size(source_root),
        "source_needed_bytes": sum(path.stat().st_size for path in source_files),
        "destination_root": destination_root,
        "staged_files": staged_files,
        "staged_bytes": sum(path.stat().st_size for path in staged_files),
        "quantization_max_abs_error": quantization_errors,
    }


def stage_mvpa(output_root: Path, session: int) -> dict[str, Any]:
    """Stage one complete Haxby session with all categories and rest volumes."""
    dataset = fetch_haxby(subjects=[2], verbose=1)
    source_bold = Path(dataset.func[0])
    source_labels = Path(dataset.session_target[0])
    downloaded_files = {
        Path(path)
        for field in (
            "anat",
            "func",
            "session_target",
            "mask_vt",
            "mask_face",
            "mask_house",
            "mask_face_little",
            "mask_house_little",
            "mask",
        )
        for path in (
            dataset[field] if isinstance(dataset[field], list) else [dataset[field]]
        )
    }
    destination_root = output_root / "mvpa"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True)

    labels = pd.read_csv(source_labels, sep=r"\s+")
    if session not in set(labels["chunks"]):
        valid = sorted(labels["chunks"].unique())
        raise ValueError(
            f"--haxby-session={session} is not present; choose from {valid}"
        )
    keep = labels["chunks"].to_numpy() == session
    trimmed_labels = labels.loc[keep].reset_index(drop=True)
    expected_categories = set(labels.loc[labels["labels"] != "rest", "labels"])
    kept_categories = set(
        trimmed_labels.loc[trimmed_labels["labels"] != "rest", "labels"]
    )
    if kept_categories != expected_categories or "rest" not in set(
        trimmed_labels["labels"]
    ):
        raise ValueError(
            f"session {session} does not preserve all categories plus rest"
        )

    source_image = nib.load(source_bold)
    data = np.asanyarray(source_image.dataobj)[..., keep]
    bold_destination = save_like(source_image, data, destination_root / "bold.nii.gz")
    labels_destination = destination_root / "labels.txt"
    trimmed_labels.to_csv(labels_destination, sep=" ", index=False)

    staged_files = [bold_destination, labels_destination]
    source_files = [source_bold, source_labels]
    return {
        "source_root": source_bold.parent.parent,
        "source_dataset_bytes": sum(path.stat().st_size for path in downloaded_files),
        "source_needed_bytes": sum(path.stat().st_size for path in source_files),
        "destination_root": destination_root,
        "staged_files": staged_files,
        "staged_bytes": sum(path.stat().st_size for path in staged_files),
        "kept_volumes": int(keep.sum()),
        "source_volumes": int(len(keep)),
        "category_counts": trimmed_labels["labels"]
        .value_counts()
        .sort_index()
        .to_dict(),
    }


def stage_encoding(output_root: Path, n_runs: int) -> dict[str, Any]:
    """Stage the first ``n_runs`` of Miyawaki2008 for the encoding tutorial.

    Encoding pairs every volume with a 10x10 stimulus, so all volumes per run are
    kept (no temporal trim); each run's 4D func is quantized to scaled int16.
    Stimulus labels and the subject-native mask are copied verbatim. Mirrors the
    notebook's ``load_runs(n_runs)``: ``func[i]``/``label[i]`` for the first runs
    plus ``mask``.
    """
    dataset = fetch_miyawaki2008(verbose=1)
    destination_root = output_root / "encoding"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True)

    source_files: list[Path] = []
    staged_files: list[Path] = []
    quantization_errors: dict[str, float] = {}

    for i in range(n_runs):
        func = Path(dataset.func[i])
        label = Path(dataset.label[i])
        source_files.extend((func, label))
        image = nib.load(func)
        if image.ndim != 4:
            raise ValueError(f"run {i}: expected 4D func, got shape {image.shape}")
        run = f"run-{i + 1:02d}"
        bold_destination = destination_root / f"{run}_bold.nii.gz"
        _, max_error = save_scaled_int16(func, image.shape[3], bold_destination)
        staged_files.append(bold_destination)
        quantization_errors[run] = max_error
        staged_files.append(copy_file(label, destination_root / f"{run}_label.csv"))

    mask = Path(dataset.mask)
    source_files.append(mask)
    staged_files.append(copy_file(mask, destination_root / "mask.nii.gz"))

    source_root = Path(dataset.func[0]).parent
    return {
        "source_root": source_root,
        "source_dataset_bytes": byte_size(source_root),
        "source_needed_bytes": sum(path.stat().st_size for path in source_files),
        "destination_root": destination_root,
        "staged_files": staged_files,
        "staged_bytes": sum(path.stat().st_size for path in staged_files),
        "kept_runs": n_runs,
        "quantization_max_abs_error": quantization_errors,
    }


def stage_isc(output_root: Path, n_subjects: int) -> dict[str, Any]:
    """Stage ``n_subjects`` of development_fmri func for the ISC tutorial.

    ISC correlates full timecourses across subjects, so each subject's timeseries
    is kept at full length (no temporal trim). The func is copied verbatim rather
    than int16-quantized: the source is already compact (~6 MB/subject) and int16
    re-encoding both grows the gzip size and would perturb the exact timeseries
    values ISC correlates. Mirrors the notebook's
    ``fetch_development_fmri(n_subjects)`` then ``DATA.func[i]`` (confounds are
    unused by the tutorial, so not staged).
    """
    dataset = fetch_development_fmri(n_subjects=n_subjects, verbose=1)
    destination_root = output_root / "isc"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True)

    source_files = [Path(path) for path in dataset.func]
    staged_files: list[Path] = []

    for i, func in enumerate(source_files, start=1):
        image = nib.load(func)
        if image.ndim != 4:
            raise ValueError(f"sub-{i:02d}: expected 4D func, got shape {image.shape}")
        destination = destination_root / f"sub-{i:02d}_task-pixar_bold.nii.gz"
        staged_files.append(copy_file(func, destination))

    source_root = source_files[0].parent.parent
    return {
        "source_root": source_root,
        "source_dataset_bytes": byte_size(source_root),
        "source_needed_bytes": sum(path.stat().st_size for path in source_files),
        "destination_root": destination_root,
        "staged_files": staged_files,
        "staged_bytes": sum(path.stat().st_size for path in staged_files),
        "kept_subjects": n_subjects,
    }


def print_report(name: str, report: dict[str, Any]) -> None:
    """Print a compact, machine-readable-enough staging report."""
    print(f"\n{name.upper()}")
    print(f"  source: {report['source_root']}")
    print(f"  full dataset bytes: {report['source_dataset_bytes']}")
    print(f"  tutorial-needed source bytes: {report['source_needed_bytes']}")
    print(f"  staged bytes: {report['staged_bytes']}")
    for path in report["staged_files"]:
        relative = path.relative_to(report["destination_root"])
        print(f"  {path.stat().st_size:>10}  tutorials/{name}/{relative.as_posix()}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        choices=("all", "glm", "mvpa", "encoding", "isc"),
        default="all",
        help="dataset to stage (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"staging root (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--glm-volumes",
        type=int,
        default=76,
        help="leading volumes to retain per GLM subject (default: 76; four complete blocks per condition)",
    )
    parser.add_argument(
        "--haxby-session",
        type=int,
        default=0,
        help="complete Haxby session/chunk to retain (default: 0)",
    )
    parser.add_argument(
        "--miyawaki-runs",
        type=int,
        default=8,
        help="leading Miyawaki runs to retain for the encoding tutorial (default: 8)",
    )
    parser.add_argument(
        "--isc-subjects",
        type=int,
        default=12,
        help="development_fmri subjects to retain for the ISC tutorial (default: 12)",
    )
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if args.dataset in ("all", "glm"):
        print_report("glm", stage_glm(output_root, args.glm_volumes))
    if args.dataset in ("all", "mvpa"):
        print_report("mvpa", stage_mvpa(output_root, args.haxby_session))
    if args.dataset in ("all", "encoding"):
        print_report("encoding", stage_encoding(output_root, args.miyawaki_runs))
    if args.dataset in ("all", "isc"):
        print_report("isc", stage_isc(output_root, args.isc_subjects))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
