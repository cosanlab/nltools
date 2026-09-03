#!/usr/bin/env python3
"""Generate API documentation from Python source with griffe2md.

Loads nltools once with griffe and renders every page in PAGES in-process
(`griffe2md.render_object_docs`), postprocesses the Markdown (see
`postprocess_api_docs`) and writes it under docs/api/. Three kinds of page:

- **module pages** — one module or class rendered with griffe2md's own template
  (the four data classes, `models`, `results`, and the internal-module
  reference);
- **task pages** (docs/api/tasks/) — a hand-written intro over an explicit list
  of objects drawn from several modules, grouped by what a user wants to do;
- **the namespace page** — every public name of ``nltools.algorithms``, A-Z.

The output filenames match the TOC entries in docs/myst.yml (a test keeps the
two in sync, and checks that every public name is documented on a page a user
will browse).

griffe warnings (missing annotations, unresolved references, ...) are always
surfaced — deduplicated, grouped by source file, with a count — because griffe
only logs them.

Usage:
    python scripts/build_api_docs.py                     # generate all
    python scripts/build_api_docs.py --clean             # rm docs/api/**/*.md first
    python scripts/build_api_docs.py --check             # drift gate: regenerate to a
                                                         # temp dir, diff vs docs/api
    python scripts/build_api_docs.py --fail-on-warnings  # exit 1 on any griffe warning
"""

from __future__ import annotations

import argparse
import io
import logging
import re
import sys
import tempfile
import tomllib
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import griffe2md
from griffe import GriffeLoader, Object, Parser

# Make the sibling postprocess module importable whether run as a script
# (sys.path[0] already covers it) or imported some other way.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from postprocess_api_docs import (  # noqa: E402
    _myst_slug,
    page_label,
    page_prefix,
    page_title,
    postprocess,
    with_frontmatter,
    xref_entries,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_API = PROJECT_ROOT / "docs" / "api"

# Heading depth of a member (``### `name```) on a task or namespace page; the
# ``## Classes`` / ``## Functions`` category headings sit one level above, so
# the layout matches a module page once its ``#`` root heading is stripped.
MEMBER_HEADING_LEVEL = 3

# Longest attribute value shown in a summary table's Type column when the
# attribute has no annotation (a ``Literal[...]`` alias fits; a registry
# dict does not).
MAX_INLINE_VALUE = 80


@dataclass(frozen=True)
class Page:
    """One generated page under docs/api.

    Exactly one of ``module`` (render that module/class with griffe2md's
    template), ``namespace`` (render every public name of that package, A-Z)
    or ``objects`` (render these public dotpaths in order) is set. ``intro`` is
    hand-written Markdown placed under the title (the page's H1 comes from the
    frontmatter ``title``). ``internal`` pages sit under "Internal modules" in
    the TOC and don't count as documentation a user is expected to browse.
    """

    output: str
    title: str
    module: str | None = None
    namespace: str | None = None
    objects: tuple[str, ...] = ()
    intro: str = ""
    internal: bool = False


def _module_page(
    module: str, output: str, *, title: str | None = None, internal: bool = False
) -> Page:
    return Page(output, title or page_title(module), module=module, internal=internal)


def _task_page(stem: str, title: str, intro: str, objects: Sequence[str]) -> Page:
    return Page(f"tasks/{stem}.md", title, objects=tuple(objects), intro=intro)


def _algorithms(*names: str) -> list[str]:
    return [f"nltools.algorithms.{n}" for n in names]


def _plotting(*names: str) -> list[str]:
    return [f"nltools.plotting.{n}" for n in names]


# Order matters twice: it is the order objects appear on a task page, and pages
# earlier in the registry win when several document the same symbol — type
# annotations then link to the task page rather than the A-Z index or an
# internal module page (see `build`).
PAGES: tuple[Page, ...] = (
    # --- data classes -------------------------------------------------------
    _module_page("nltools.data.braindata.BrainData", "data/brain_data.md"),
    _module_page("nltools.data.adjacency.Adjacency", "data/adjacency.md"),
    _module_page("nltools.data.designmatrix.DesignMatrix", "data/design_matrix.md"),
    _module_page("nltools.data.collection.BrainCollection", "data/brain_collection.md"),
    _module_page("nltools.data.results", "data/results.md"),
    _module_page("nltools.models", "models.md"),
    # --- functions by task --------------------------------------------------
    _task_page(
        "loading",
        "Loading, masks & datasets",
        "Get data into nltools. [BrainData](../data/brain_data.md) and the other "
        "data classes load NIfTI files and HDF5 bundles themselves. The functions "
        "here cover the rest: example datasets and Neurovault collections, sphere "
        "and ROI masks, `concatenate`, and the MNI template every object falls back "
        "on when it gets no mask (`set_brainspace`).",
        [
            "nltools.io.load_brain_data_h5",
            "nltools.io.to_h5",
            "nltools.io.is_h5_path",
            "nltools.datasets.fetch_pain",
            "nltools.datasets.fetch_emotion_ratings",
            "nltools.datasets.fetch_neurovault_collection",
            "nltools.datasets.load_haxby_example",
            "nltools.datasets.download_nifti",
            "nltools.mask.create_sphere",
            "nltools.mask.expand_mask",
            "nltools.mask.collapse_mask",
            "nltools.mask.roi_to_brain",
            "nltools.mask.roi_to_brain_from_atlas",
            "nltools.utils.concatenate",
            "nltools.templates.BrainSpaceConfig",
            "nltools.templates.get_brainspace",
            "nltools.templates.set_brainspace",
            "nltools.templates.reset_brainspace",
            "nltools.templates.with_brainspace",
            "nltools.templates.fetch_resource",
            "nltools.templates.list_resources",
            "nltools.templates.get_bg_image",
            "nltools.templates.is_standard_space",
            "nltools.templates.detect_resolution",
        ],
    ),
    _task_page(
        "preprocessing",
        "Preprocessing & signal",
        "Clean timeseries before modelling. Standardize or trim outliers, flag "
        "motion spikes, resample to another sampling rate, build cosine drift "
        "regressors. Every function takes and returns numpy arrays or DataFrames. "
        "The [BrainData](../data/brain_data.md) and "
        "[DesignMatrix](../data/design_matrix.md) methods of the same name call "
        "them.",
        _algorithms(
            "zscore",
            "trim",
            "winsorize",
            "find_spikes",
            "downsample",
            "upsample",
            "make_cosine_basis",
            "calc_bpm",
        ),
    ),
    _task_page(
        "design-and-glm",
        "Design matrices, HRF & GLM",
        "Build a first-level model. `events_to_dm` turns an events table into a "
        "[DesignMatrix](../data/design_matrix.md); the HRF functions sample the SPM "
        "and Glover responses and their derivatives for convolution. `regress` is "
        "the standalone numpy GLM. For 4D data use `BrainData.fit(model='glm')`, "
        "which raises the warning classes listed here when a design is "
        "rank-deficient or nearly collinear.",
        [
            "nltools.data.designmatrix.io.events_to_dm",
            *_algorithms(
                "spm_hrf",
                "spm_time_derivative",
                "spm_dispersion_derivative",
                "glover_hrf",
                "glover_time_derivative",
                "glover_dispersion_derivative",
                "regress",
            ),
            "nltools.data.braindata.modeling.RankDeficientDesignWarning",
            "nltools.data.braindata.modeling.NearCollinearDesignWarning",
            "nltools.utils.DesignMatrixWarning",
            "nltools.utils.ResamplingWarning",
        ],
    ),
    _task_page(
        "prediction",
        "Prediction & cross-validation",
        "Decode or predict from brain data. `BrainData.predict` and "
        "`BrainCollection.predict_group` run the workflow. Listed here are the "
        "pieces they accept or return: the cross-validation schemes (`resolve_cv` "
        "turns an int, a name, or an sklearn splitter into one), the ridge solvers "
        "behind `model='ridge'` (CPU or GPU), `Roc` for a classifier's output, and "
        "the plots of weights, margins, and predictions.",
        [
            "nltools.cross_validation.KFoldStratified",
            "nltools.cross_validation.resolve_cv",
            *_algorithms("ridge_cv", "ridge_svd"),
            "nltools.data.roc.Roc",
            *_plotting(
                "plot_roc",
                "plot_dist_from_hyperplane",
                "plot_probability",
                "plot_scatter",
            ),
        ],
    ),
    _task_page(
        "similarity",
        "Similarity & RSA",
        "Compare patterns and matrices. `compute_similarity` scores two arrays "
        "under a `metric=`; the Fisher transforms make correlations averageable. "
        "`matrix_permutation_test` (Mantel), `correlation_permutation_test`, and "
        "`distance_correlation` compare whole matrices. The plots summarize stacks "
        "of [Adjacency](../data/adjacency.md) matrices, and `SpatialScale` records "
        "which ROI or searchlight each matrix in a stack came from so a reduction "
        "can be painted back onto the brain.",
        [
            *_algorithms(
                "compute_similarity",
                "compute_multivariate_similarity",
                "transform_pairwise",
                "fisher_r_to_z",
                "fisher_z_to_r",
                "matrix_permutation_test",
                "correlation_permutation_test",
                "distance_correlation",
                "double_center",
                "u_center",
            ),
            *_plotting(
                "plot_stacked_adjacency",
                "plot_mean_label_distance",
                "plot_between_label_distance",
                "plot_silhouette",
            ),
            "nltools.data.adjacency.spatial.SpatialScale",
        ],
    ),
    _task_page(
        "alignment",
        "Functional alignment",
        "Put subjects into a shared functional space. `align` is the whole-brain "
        "entry point, with `method='procrustes'` or an SRM variant. `SRM`, `DetSRM`, "
        "`HyperAlignment`, and `LocalAlignment` are the sklearn-style estimators; "
        "`LocalAlignment` fits one transform per ROI or searchlight. `align_states` "
        "matches state maps across groups. `BrainData.align` calls `align`.",
        _algorithms(
            "align",
            "align_states",
            "procrustes",
            "procrustes_distance",
            "SRM",
            "DetSRM",
            "HyperAlignment",
            "LocalAlignment",
        ),
    ),
    _task_page(
        "inference",
        "Statistics & inference",
        "Non-parametric group statistics. The one-sample, two-sample, and "
        "timeseries permutation tests run on CPU or GPU (`device=`); "
        "`phase_randomize` and `circle_shift` are the timeseries null models. "
        "`OnlineBootstrapStats` keeps running mean and variance over bootstrap "
        "draws instead of storing them. `fdr`, `holm_bonf`, `threshold`, and "
        "`multi_threshold` correct or threshold the resulting p-maps. "
        "`BrainData.ttest`, `Adjacency.ttest`, and `BrainData.bootstrap` call "
        "these.",
        [
            *_algorithms(
                "one_sample_permutation_test",
                "two_sample_permutation_test",
                "timeseries_correlation_permutation_test",
                "phase_randomize",
                "circle_shift",
            ),
            "nltools.algorithms.inference.OnlineBootstrapStats",
            *_algorithms("fdr", "holm_bonf", "threshold", "multi_threshold"),
        ],
    ),
    _task_page(
        "intersubject",
        "Intersubject correlation",
        "Measure time-locked responses shared across subjects. `isc` correlates "
        "each subject's timeseries with the rest of the group and bootstraps a "
        "confidence interval. `isfc` does the same across regions, `isps` measures "
        "phase synchrony, and `isc_group` compares two groups by permutation. The "
        "two `*_permutation_test` functions are the CPU/GPU engines (`device=`) "
        "underneath, shared with the [permutation tests](inference.md).",
        _algorithms(
            "isc",
            "isfc",
            "isps",
            "isc_group",
            "isc_permutation_test",
            "isc_group_permutation_test",
        ),
    ),
    _task_page(
        "plotting",
        "Brain plotting",
        "Render a volume on the cortical surface, as a flatmap, or in an "
        "interactive viewer, and browse ICA/PCA components. `BrainData.plot` and "
        "`BrainData.iplot` call these. Plots of model output sit with their "
        "workflow. ROC and prediction plots are under "
        "[Prediction & cross-validation](prediction.md); adjacency-matrix plots "
        "are under [Similarity & RSA](similarity.md).",
        _plotting(
            "plot_surf", "plot_flatmap", "plot_interactive_brain", "component_viewer"
        ),
    ),
    _task_page(
        "atlases",
        "Atlases & cluster reports",
        "Put anatomical names on a result. `list_atlases` and `load_atlas` fetch "
        "parcellations from the nltools Hugging Face dataset on first use, and "
        "`label_coords` looks MNI coordinates up in them. `BrainData.cluster_report` "
        "(`cluster_report_data` underneath) thresholds a statistical map and labels "
        "each cluster's peak. `roi_to_brain_from_atlas` paints per-parcel values "
        "back into a volume.",
        [
            "nltools.data.atlases.list_atlases",
            "nltools.data.atlases.load_atlas",
            "nltools.data.atlases.Atlas",
            "nltools.data.atlases.AtlasMetadata",
            "nltools.data.atlases.AtlasKind",
            "nltools.data.atlases.ATLASES",
            "nltools.data.atlases.DEFAULT_ATLASES",
            "nltools.data.atlases.label_coords",
            "nltools.data.atlases.ClusterReport",
            "nltools.data.atlases.cluster_report_data",
            "nltools.mask.roi_to_brain_from_atlas",
        ],
    ),
    _task_page(
        "simulation",
        "Simulation",
        "Synthetic data with a known signal, for testing a pipeline end to end. "
        "`Simulator` builds [BrainData](../data/brain_data.md) with Gaussian-blob "
        "signal, several subjects, and noise. `SimulateGrid` builds 2D grids, "
        "which is enough to exercise thresholding and multiple-comparison "
        "correction without a mask.",
        ["nltools.data.simulator.Simulator", "nltools.data.simulator.SimulateGrid"],
    ),
    # --- the flat namespace, A-Z --------------------------------------------
    Page(
        "algorithms.md",
        "nltools.algorithms (A–Z index)",
        namespace="nltools.algorithms",
        intro=(
            "Every public function and class of `nltools.algorithms`, alphabetically. "
            "Use this page when you know the name; the *Functions by task* pages "
            "group the same objects by what they are for."
        ),
    ),
    # --- internal modules ---------------------------------------------------
    _module_page("nltools.algorithms.backends", "backends.md", internal=True),
    _module_page("nltools.data.braindata.cache", "cache.md", internal=True),
    _module_page(
        "nltools.data.braindata.neighborhoods", "neighborhoods.md", internal=True
    ),
    _module_page("nltools.utils", "utils.md", internal=True),
    _module_page("nltools.templates", "templates.md", internal=True),
    _module_page("nltools.algorithms.ridge", "algorithms/ridge.md", internal=True),
    _module_page(
        "nltools.algorithms.inference", "algorithms/inference.md", internal=True
    ),
    _module_page("nltools.data.braindata.io", "data/braindata_io.md", internal=True),
    _module_page(
        "nltools.data.braindata.analysis", "data/braindata_analysis.md", internal=True
    ),
    _module_page(
        "nltools.data.braindata.modeling", "data/braindata_modeling.md", internal=True
    ),
    _module_page(
        "nltools.data.braindata.prediction",
        "data/braindata_prediction.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.braindata.bootstrap",
        "data/braindata_bootstrap.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.braindata.plotting", "data/braindata_plotting.md", internal=True
    ),
    _module_page(
        "nltools.data.adjacency.stats", "data/adjacency_stats.md", internal=True
    ),
    _module_page(
        "nltools.data.adjacency.modeling", "data/adjacency_modeling.md", internal=True
    ),
    _module_page(
        "nltools.data.adjacency.plotting", "data/adjacency_plotting.md", internal=True
    ),
    _module_page("nltools.data.adjacency.io", "data/adjacency_io.md", internal=True),
    _module_page(
        "nltools.data.adjacency.spatial", "data/adjacency_spatial.md", internal=True
    ),
    _module_page(
        "nltools.data.designmatrix.transforms",
        "data/design_matrix_transforms.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.designmatrix.regressors",
        "data/design_matrix_regressors.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.designmatrix.append",
        "data/design_matrix_append.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.designmatrix.diagnostics",
        "data/design_matrix_diagnostics.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.designmatrix.plotting",
        "data/design_matrix_plotting.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.designmatrix.io", "data/design_matrix_io.md", internal=True
    ),
    _module_page(
        "nltools.data.collection.core", "data/collection_core.md", internal=True
    ),
    _module_page(
        "nltools.data.collection.execution",
        "data/collection_execution.md",
        internal=True,
    ),
    _module_page(
        "nltools.data.collection.inference",
        "data/collection_inference.md",
        internal=True,
    ),
    _module_page("nltools.data.collection.io", "data/collection_io.md", internal=True),
)


# ---------------------------------------------------------------------------
# griffe warnings
# ---------------------------------------------------------------------------

# ``path/to/file.py:123: message`` — griffe's warning format.
_WARNING_RE = re.compile(r"^(?P<file>[^\s:]+):(?P<line>\d+): (?P<msg>.+)$")


def parse_warnings(stderr: str) -> list[str]:
    """Return griffe's captured log output as a list of warning lines (blanks dropped).

    Nothing is filtered: "No type or annotation" lines are real docstring bugs.
    """
    return [ln.rstrip() for ln in stderr.splitlines() if ln.strip()]


@contextmanager
def capture_griffe_warnings() -> Iterator[io.StringIO]:
    """Collect everything the ``griffe`` logger emits at WARNING or above.

    griffe reports docstring problems (``file:line: message``) through logging
    while loading and, lazily, while docstrings are first parsed during
    rendering. The CLI printed them to stderr; in-process we attach a handler
    for the duration of a build and read the stream afterwards.
    """
    logger = logging.getLogger("griffe")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter("%(message)s"))
    previous_level = logger.level
    if logger.getEffectiveLevel() > logging.WARNING:
        logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    try:
        yield stream
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


def warning_report(warnings: set[str] | list[str]) -> str:
    """Format warnings deduplicated, grouped by source file, with a count summary.

    The same object rendered on several pages (a facade class and its own page)
    repeats its warnings verbatim; each appears once here. Lines that don't match
    griffe's ``file:line: message`` format are listed under ``(other)``.
    """
    unique = sorted(set(warnings))
    if not unique:
        return "0 griffe warnings"
    by_file: dict[str, list[str]] = defaultdict(list)
    for w in unique:
        m = _WARNING_RE.match(w)
        if m:
            by_file[m["file"]].append(f"{m['line']}: {m['msg']}")
        else:
            by_file["(other)"].append(w)
    out: list[str] = []
    for file in sorted(by_file):
        out.append(file)
        out.extend(f"  {entry}" for entry in by_file[file])
    n_files = len(by_file)
    out.append(
        f"{len(unique)} griffe warning{'s' if len(unique) != 1 else ''} "
        f"in {n_files} file{'s' if n_files != 1 else ''}"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    """What one build produced: the pages, failures, and every griffe warning seen."""

    ok: int = 0
    failed: list[str] = field(default_factory=list)
    warnings: set[str] = field(default_factory=set)
    pages: dict[str, str] = field(default_factory=dict)  # output path -> final text


def load_griffe2md_config() -> dict[str, Any]:
    """The ``[tool.griffe2md]`` table from pyproject.toml (path-independent)."""
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["tool"]["griffe2md"]


class ApiIndex:
    """griffe's view of nltools, loaded once, addressed by public dotted path.

    Mirrors what the griffe2md CLI did per invocation (same parser, docstring
    options and search path) so in-process rendering is byte-identical, and
    resolves the aliases a facade re-export creates (``nltools.algorithms.fdr``
    -> the function defined in ``nltools.algorithms.corrections``).
    """

    def __init__(self, config: dict[str, Any]) -> None:
        loader = GriffeLoader(
            docstring_parser=Parser(config["docstring_style"]),
            docstring_options=config["docstring_options"],
            search_paths=[str(PROJECT_ROOT), *sys.path],
        )
        self.package = loader.load("nltools")
        loader.resolve_aliases(external=True)

    def resolve(self, dotpath: str) -> Object:
        """The object a public dotted path names, aliases followed to their target."""
        rel = dotpath.removeprefix("nltools.")
        obj = self.package if rel in ("", "nltools") else self.package[rel]
        target = obj.final_target if obj.is_alias else obj
        # A function shadowed by its same-named submodule: to griffe,
        # ``nltools.algorithms.procrustes`` *is* the module
        # ``algorithms.alignment.procrustes`` (the submodule wins over the
        # ``from .procrustes import procrustes`` alias). The caller meant the
        # function inside it — the module isn't a documentable page member.
        if target.is_module and target.name in target.members:
            inner = target.members[target.name]
            if not inner.is_module:
                target = inner.final_target if inner.is_alias else inner
        return target

    def public_names(self, namespace: str) -> list[str]:
        """The namespace's ``__all__``, sorted, minus submodules it re-exports."""
        module = self.resolve(namespace)
        return sorted(
            (
                name
                for name in module.exports or ()
                if not self.resolve(f"{namespace}.{name}").is_module
            ),
            key=str.lower,
        )


@dataclass(frozen=True)
class Member:
    """One object rendered onto a task or namespace page.

    ``body`` is griffe2md's Markdown for the object at `MEMBER_HEADING_LEVEL`
    (empty for attributes, which appear in the summary table only, like the
    postprocess treats attributes on module pages). ``paths`` are the dotted
    paths the member documents — the public one it was listed under and, when
    different, griffe's canonical definition — so type annotations written
    either way link here.
    """

    name: str
    kind: str  # "class" | "function" | "attribute"
    description: str
    body: str
    annotation: str | None = None
    paths: tuple[str, ...] = ()


def render_member(obj: Object, public_path: str, config: dict[str, Any]) -> Member:
    """Render one griffe object as a task-page member."""
    description = obj.docstring.value.split("\n", 1)[0] if obj.docstring else ""
    is_attribute = obj.kind.value == "attribute"
    body = (
        ""
        if is_attribute
        else griffe2md.render_object_docs(
            obj, {**config, "heading_level": MEMBER_HEADING_LEVEL}
        )
    )
    annotation = None
    if is_attribute:
        # Type column: the annotation, or for an unannotated alias such as
        # ``AtlasKind = Literal[...]`` the (short) value, which *is* its type.
        if obj.annotation:
            annotation = str(obj.annotation)
        elif obj.value is not None and len(str(obj.value)) <= MAX_INLINE_VALUE:
            annotation = str(obj.value)
    return Member(
        name=obj.name,
        kind=obj.kind.value,
        description=description,
        body=body,
        annotation=annotation,
        paths=tuple(dict.fromkeys([public_path, obj.path])),
    )


_SECTION_TITLES = {
    "attribute": "Attributes",
    "class": "Classes",
    "function": "Functions",
}


def _summary_table(kind: str, members: Sequence[Member]) -> str:
    title = _SECTION_TITLES[kind]
    if kind == "attribute":
        rows = [
            f"`{m.name}` | "
            f"{f'<code>{m.annotation}</code>' if m.annotation else ''} | "
            f"{m.description}"
            for m in members
        ]
        header = "Name | Type | Description\n---- | ---- | -----------"
    else:
        rows = [
            f"[`{m.name}`](#{_myst_slug(m.name)}) | {m.description}" for m in members
        ]
        header = "Name | Description\n---- | -----------"
    return f"**{title}:**\n\n{header}\n" + "\n".join(rows)


def compose_objects_page(intro: str, members: Sequence[Member]) -> str:
    """Assemble a task page: intro, per-kind summary tables, then the members.

    Summary tables follow the postprocess's canonical order (Attributes,
    Classes, Functions); detail sections are ``## Classes`` then
    ``## Functions``, members in registry order. Attributes have no detail
    section. The result goes through `postprocess` like any griffe2md page.
    """
    by_kind = {kind: [m for m in members if m.kind == kind] for kind in _SECTION_TITLES}
    parts = [intro.strip()]
    parts.extend(
        _summary_table(kind, group) for kind, group in by_kind.items() if group
    )
    for kind in ("class", "function"):
        if by_kind[kind]:
            parts.append(f"## {_SECTION_TITLES[kind]}")
            parts.extend(m.body.strip() for m in by_kind[kind])
    return "\n\n".join(parts) + "\n"


@dataclass(frozen=True)
class RenderedPage:
    """A page's raw (pre-postprocess) Markdown plus what `xref_entries` needs."""

    page: Page
    text: str
    roots: dict[str, tuple[str, ...]] | None  # task pages: heading -> dotted paths


def render_page(page: Page, index: ApiIndex, config: dict[str, Any]) -> RenderedPage:
    """Render one registry entry to raw Markdown (griffe2md output, unprocessed)."""
    if page.module:
        text = griffe2md.render_object_docs(index.resolve(page.module), config)
        return RenderedPage(page, text, None)

    if page.namespace:
        paths = [f"{page.namespace}.{n}" for n in index.public_names(page.namespace)]
        # The package docstring opens the page, under the hand-written intro.
        docstring = griffe2md.render_object_docs(
            index.resolve(page.namespace),
            {**config, "members": False, "show_root_heading": False},
        )
        intro = f"{page.intro}\n\n{docstring.strip()}"
    else:
        paths = list(page.objects)
        intro = page.intro

    members = [render_member(index.resolve(p), p, config) for p in paths]
    roots = {m.name: m.paths for m in members}
    return RenderedPage(page, compose_objects_page(intro, members), roots)


def build(out_dir: Path, *, clean: bool = False, verbose: bool = True) -> BuildReport:
    """Generate every page in PAGES into ``out_dir`` (normally docs/api).

    Two passes over the raw Markdown: the first postprocesses each page without
    cross-page links and indexes the labels it produced (`xref_entries`); the
    second postprocesses again with that index so type annotations and
    ``Bases:`` entries link to the page documenting them. When several pages
    document one symbol, the first in PAGES wins (task pages precede the A-Z
    index and the internal modules).
    """
    report = BuildReport()
    if clean:
        for f in out_dir.rglob("*.md"):
            f.unlink()
        if verbose:
            print(f"Cleaned {out_dir}")

    config = load_griffe2md_config()
    rendered: list[RenderedPage] = []
    with capture_griffe_warnings() as stream:
        index = ApiIndex(config)
        for page in PAGES:
            source = page.module or page.namespace or f"{len(page.objects)} objects"
            if verbose:
                print(f"  {page.output} ← {source}")
            try:
                rendered.append(render_page(page, index, config))
            except Exception as exc:  # a bad dotpath in PAGES, a template error
                print(f"  FAILED: {page.output}: {exc!r}", file=sys.stderr)
                report.failed.append(page.output)
    report.warnings.update(parse_warnings(stream.getvalue()))

    xref: dict[str, str] = {}
    prefixes: dict[str, str] = {}
    for item in rendered:
        prefix = page_prefix(out_dir / item.page.output, out_dir)
        prefixes[item.page.output] = prefix
        entries = xref_entries(
            prefix,
            postprocess(item.text, prefix),
            module=item.page.module,
            roots=item.roots,
        )
        for path, label in entries.items():
            xref.setdefault(path, label)

    for item in rendered:
        output = out_dir / item.page.output
        output.parent.mkdir(parents=True, exist_ok=True)
        prefix = prefixes[item.page.output]
        body = postprocess(item.text, prefix, xref)
        text = with_frontmatter(body, item.page.title, page_label(prefix))
        output.write_text(text)
        report.pages[item.page.output] = text
        report.ok += 1
    return report


# ---------------------------------------------------------------------------
# drift check
# ---------------------------------------------------------------------------


def diff_trees(generated: Path, committed: Path) -> list[str]:
    """Compare two docs/api trees; one message per changed, stale, or new page."""
    gen = {p.relative_to(generated).as_posix(): p for p in generated.rglob("*.md")}
    com = {p.relative_to(committed).as_posix(): p for p in committed.rglob("*.md")}
    drift: list[str] = []
    for rel in sorted(gen.keys() | com.keys()):
        if rel not in com:
            drift.append(f"new (not committed): {rel}")
        elif rel not in gen:
            drift.append(f"missing (stale, delete): {rel}")
        elif gen[rel].read_text() != com[rel].read_text():
            drift.append(f"changed: {rel}")
    return drift


def check(docs_api: Path = DOCS_API) -> int:
    """Regenerate into a temp dir and diff against ``docs_api``; 0 when in sync."""
    with tempfile.TemporaryDirectory(prefix="nltools-api-docs-") as tmp:
        report = build(Path(tmp), verbose=False)
        if report.failed:
            print(f"build failed for: {', '.join(report.failed)}", file=sys.stderr)
            return 1
        drift = diff_trees(Path(tmp), docs_api)
    if drift:
        print("docs/api is out of date — run `uv run poe docs-generate` and commit:")
        for line in drift:
            print(f"  {line}")
        return 1
    print(f"docs/api in sync ({report.ok} pages)")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing generated docs before generating",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temp dir and fail if docs/api differs (no writes)",
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit non-zero if griffe emitted any warning",
    )
    args = parser.parse_args()

    if args.check:
        sys.exit(check())

    report = build(DOCS_API, clean=args.clean)
    print()
    print(warning_report(report.warnings), file=sys.stderr)
    print(f"\nGenerated {report.ok} API doc pages ({len(report.failed)} failures)")
    if report.failed or (args.fail_on_warnings and report.warnings):
        sys.exit(1)


if __name__ == "__main__":
    main()
