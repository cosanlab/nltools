"""niivue interactive viewer for BrainData, as a self-owned `anywidget`.

`build_viewer` returns a `NiivueViewer` — a WebGL brain viewer with live
windowing, slice scrolling, native 4D frame scrubbing, true 3D rendering, and
optional nltools-atlas overlays (colored regions / outlines / hover labels).

The widget drives the `@niivue/niivue` JavaScript library directly through
anywidget's standard model API (see ``viewer.js``), so it renders identically
in Jupyter and ``marimo edit`` without depending on any host-specific protocol.

The module is split functional-core / imperative-shell:

- Pure helpers (`resolve_cmap`, `divergent_partner`, `slice_type_for`,
  `qualitative_colors`, `atlas_to_label_lut`, `resolve_background`,
  `bd_to_nifti_bytes`, `compute_display_window`) translate BrainData /
  `Atlas` state into the vocabulary niivue understands.
- `NiivueViewer` is the thin traitlets widget; `build_viewer` is the assembler
  that fills its traits from a BrainData.

niivue formatting deliberately lives here, not in ``nltools/data/atlases/`` —
the atlas package stays niivue-agnostic and only exposes the generic `Atlas`
dataclass.
"""

from __future__ import annotations

import colorsys
import functools
import gzip
import pathlib
import warnings
from dataclasses import dataclass
from typing import Literal

import anywidget
import traitlets

from nltools.data.atlases import Atlas, load_atlas
from nltools.templates.matching import get_bg_image, is_standard_space
from nltools.utils import find_stack_level

_VIEWER_JS = pathlib.Path(__file__).parent / "viewer.js"


# --------------------------------------------------------------------------- #
# Colormaps
# --------------------------------------------------------------------------- #

# niivue's builtin colormap names (@niivue/niivue 0.69). Hardcoded rather than
# read from a Python package's statics because the JS engine now comes straight
# from a CDN — there is no local niivue install to introspect. Stable across
# niivue releases; extend if niivue adds builtins we want to expose by name.
_NIIVUE_COLORMAPS: frozenset[str] = frozenset(
    {
        "actc", "afni_blues_inv", "afni_reds_inv", "batlow", "bcgwhw",
        "bcgwhw_dark", "blue", "blue2cyan", "blue2magenta", "blue2red",
        "bluegrn", "bone", "bronze", "cet_l17", "cividis", "cool", "copper",
        "copper2", "ct_airways", "ct_artery", "ct_bones", "ct_brain",
        "ct_brain_gray", "ct_cardiac", "ct_head", "ct_kidneys", "ct_liver",
        "ct_muscles", "ct_scalp", "ct_skull", "ct_soft", "ct_soft_tissue",
        "ct_surface", "ct_vessels", "ct_w_contrast", "cubehelix",
        "electric_blue", "freesurfer", "ge_color", "gold", "gray", "green",
        "green2cyan", "green2orange", "hot", "hotiron", "hsv", "inferno",
        "jet", "kry", "linspecer", "lipari", "magma", "mako", "navia", "nih",
        "plasma", "random", "red", "redyell", "rocket", "roi_i256", "surface",
        "thermal", "turbo", "violet", "viridis", "warm", "winter", "x_rain",
    }
)  # fmt: skip

# Common matplotlib colormap names with no exact niivue equivalent. niivue
# silently renders gray for unknown names, so we map the popular ones and
# warn (see resolve_cmap) rather than letting them fall through.
_MPL_TO_NIIVUE: dict[str, str] = {
    "rdbu_r": "warm",
    "rdbu": "winter",
    "coolwarm": "warm",
    "bwr": "warm",
    "seismic": "warm",
    "spectral": "warm",
    "spectral_r": "warm",
    "reds": "warm",
    "reds_r": "warm",
    "oranges": "warm",
    "ylorrd": "redyell",
    "blues": "winter",
    "blues_r": "winter",
    "greens": "green",
    "purples": "violet",
    "greys": "gray",
    "grays": "gray",
    "grey": "gray",
}

# Cool-side partner for the positive colormap, used as ``colormap_negative``
# so divergent stat maps render with a mirrored negative limb. Falls back to
# ``winter`` for sequential maps with no obvious counterpart.
_DIVERGENT_PARTNERS: dict[str, str] = {
    "warm": "winter",
    "hot": "cool",
    "hotiron": "cool",
    "red": "blue",
    "redyell": "blue",
    "gold": "blue",
    "green": "violet",
    "viridis": "winter",
    "inferno": "winter",
    "magma": "winter",
    "plasma": "winter",
    "jet": "winter",
}


@functools.cache
def _niivue_colormaps() -> frozenset[str]:
    """Names of niivue's builtin colormaps (see `_NIIVUE_COLORMAPS`)."""
    return _NIIVUE_COLORMAPS


def resolve_cmap(name: str) -> str:
    """Resolve a colormap name to a valid niivue colormap.

    Valid niivue names pass through. Common matplotlib names are mapped to
    the closest niivue equivalent (with a warning, since the mapping is
    lossy). Anything else falls back to ``"warm"`` with a warning, because
    niivue renders unknown colormaps as flat gray with no error.

    Args:
        name: A niivue or matplotlib colormap name (case-insensitive).

    Returns:
        A valid niivue colormap name.
    """
    key = name.lower()
    if key in _niivue_colormaps():
        return key
    if key in _MPL_TO_NIIVUE:
        mapped = _MPL_TO_NIIVUE[key]
        warnings.warn(
            f"colormap {name!r} is a matplotlib name with no exact niivue "
            f"equivalent; using {mapped!r}. Pass a niivue colormap name to "
            "silence this.",
            UserWarning,
            stacklevel=find_stack_level(),
        )
        return mapped
    warnings.warn(
        f"colormap {name!r} is not a known niivue colormap; falling back to 'warm'.",
        UserWarning,
        stacklevel=find_stack_level(),
    )
    return "warm"


def divergent_partner(cmap: str) -> str:
    """Return the ``colormap_negative`` partner for a positive colormap.

    Args:
        cmap: A (resolved) niivue positive colormap name.

    Returns:
        The niivue colormap to use for negative values.
    """
    return _DIVERGENT_PARTNERS.get(cmap, "winter")


# --------------------------------------------------------------------------- #
# View / slice type
# --------------------------------------------------------------------------- #

# Map a ``view`` string to the name of niivue's ``SLICE_TYPE`` enum member,
# which ``viewer.js`` indexes into (``SLICE_TYPE[name]``).
_VIEW_TO_SLICE: dict[str, str] = {
    "ortho": "MULTIPLANAR",
    "axial": "AXIAL",
    "coronal": "CORONAL",
    "sagittal": "SAGITTAL",
    "render": "RENDER",
}


def slice_type_for(view: str) -> str:
    """Map a ``view`` string to a niivue ``SLICE_TYPE`` enum name.

    Args:
        view: One of ``"ortho"``, ``"axial"``, ``"coronal"``,
            ``"sagittal"``, ``"render"``.

    Returns:
        str: The matching ``SLICE_TYPE`` member name (e.g. ``"MULTIPLANAR"``),
            which ``viewer.js`` resolves against niivue's enum.

    Raises:
        ValueError: For ``view="surface"`` (dropped — niivue's 3D render is
            volumetric, not a cortical mesh) or any other unknown view.
    """
    if view == "surface":
        raise ValueError(
            "view='surface' is no longer supported: niivue's 3D mode renders "
            "the volume, not a cortical mesh. Use view='render' for a 3D "
            "volume render, or BrainData.plot_flatmap()/plot_surf() for a "
            "surface projection."
        )
    try:
        return _VIEW_TO_SLICE[view]
    except KeyError:
        raise ValueError(
            f"view={view!r} not recognized; choose from {sorted(_VIEW_TO_SLICE)}."
        ) from None


# --------------------------------------------------------------------------- #
# Atlas overlay
# --------------------------------------------------------------------------- #


def qualitative_colors(n: int, *, seed: int = 0) -> list[tuple[int, int, int]]:
    """Deterministic qualitative RGB palette of length ``n``.

    Hues are spaced by the golden angle for maximal separation; saturation
    and value cycle through three bands so adjacent indices stay visually
    distinct. Atlases carry no color data, so this assigns region colors.

    Args:
        n: Number of colors to generate.
        seed: Rotates the starting hue; deterministic for a given seed.

    Returns:
        ``n`` ``(r, g, b)`` tuples with components in ``0..255``.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    golden = 0.6180339887498949
    h0 = (seed * golden) % 1.0
    sat_bands = (0.65, 0.85, 0.75)
    val_bands = (0.95, 0.80, 0.90)
    out: list[tuple[int, int, int]] = []
    for i in range(n):
        h = (h0 + i * golden) % 1.0
        s = sat_bands[i % 3]
        v = val_bands[i % 3]
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append((round(r * 255), round(g * 255), round(b * 255)))
    return out


def atlas_to_label_lut(atlas: Atlas) -> dict:
    """Build a niivue integer-indexed label LUT from a deterministic atlas.

    The LUT arrays are dense (length ``max_index + 1``) because niivue
    indexes them by integer voxel value. Index 0 and any gap indices are
    transparent (``A=0``, empty label); each present region gets a color
    from `qualitative_colors` (assigned in table-enumeration order, so
    colors stay stable under sparse / non-contiguous indices) and its name.

    Args:
        atlas: A loaded deterministic `Atlas`.

    Returns:
        dict: Keys ``"R"``, ``"G"``, ``"B"``, ``"A"``, ``"labels"``, suitable for
            niivue's ``setColormapLabel``.

    Raises:
        ValueError: If ``atlas`` is probabilistic (4D) — threshold it to a
            label image first.
    """
    if atlas.kind == "probabilistic":
        raise ValueError(
            f"atlas overlay supports deterministic atlases only; "
            f"{atlas.name!r} is probabilistic (4D). Threshold to a label "
            "image first."
        )
    rows = zip(atlas.labels["index"].to_list(), atlas.labels["name"].to_list())
    present = [(int(idx), str(name)) for idx, name in rows if int(idx) > 0]
    max_index = max((idx for idx, _ in present), default=0)
    size = max_index + 1

    r = [0] * size
    g = [0] * size
    b = [0] * size
    a = [0] * size
    labels = [""] * size
    for (idx, name), (cr, cg, cb) in zip(present, qualitative_colors(len(present))):
        r[idx], g[idx], b[idx], a[idx] = cr, cg, cb, 255
        labels[idx] = name
    return {"R": r, "G": g, "B": b, "A": a, "labels": labels}


def _coerce_atlas(atlas: str | Atlas | None) -> Atlas | None:
    """Resolve the ``atlas`` argument to an `Atlas` or ``None``."""
    if atlas is None:
        return None
    if isinstance(atlas, Atlas):
        return atlas
    if isinstance(atlas, str):
        return load_atlas(atlas)
    raise TypeError(
        f"atlas must be a str name, an Atlas, or None; got {type(atlas).__name__}."
    )


# --------------------------------------------------------------------------- #
# Background
# --------------------------------------------------------------------------- #


def resolve_background(affine, bg_img: str | bool | None) -> str | None:
    """Resolve the ``bg_img`` argument to a background-image path or ``None``.

    Args:
        affine (np.ndarray): 4x4 affine of the BrainData (``bd.mask.affine``),
            used to decide whether auto-MNI applies.
        bg_img: ``False`` → no background; a string/path → used as-is;
            ``None``/``True`` (auto) → the matching MNI template when the
            affine is standard space, else ``None``.

    Returns:
        A path to a background image, or ``None`` for no background.

    Note:
        ``is_standard_space(np.eye(4)) == (True, None)`` (1mm is a valid
        template resolution), so identity-affine fixtures count as standard
        space and auto would fetch a template from HuggingFace. Offline
        callers should pass ``bg_img=False``.
    """
    if bg_img is False:
        return None
    if bg_img is None or bg_img is True:
        ok, _ = is_standard_space(affine)
        return get_bg_image(affine) if ok else None
    return str(bg_img)


# --------------------------------------------------------------------------- #
# Volume bytes
# --------------------------------------------------------------------------- #


def gzip_nifti(raw: bytes) -> bytes:
    """Gzip NIfTI bytes unless they are already gzip-compressed.

    The frontend hands every volume to niivue under a ``.nii.gz`` name, so the
    payload must actually be gzip. Compressing also shrinks the single ``Bytes``
    trait that crosses the anywidget comm (a full 1mm-FOV volume is tens of MB
    raw). Bytes already carrying the gzip magic (``1f 8b`` — e.g. a template
    read straight off disk) pass through untouched.

    Args:
        raw: NIfTI-1 bytes, compressed or not.

    Returns:
        Gzip-compressed NIfTI bytes.
    """
    # mtime=0: gzip otherwise stamps the wall clock into the header, so the
    # same volume serialized twice yields different bytes. Static-site
    # builders (marimo-book) content-address these buffers and re-export a
    # notebook once per slider value; a deterministic stream lets identical
    # volumes de-duplicate to one file.
    return raw if raw[:2] == b"\x1f\x8b" else gzip.compress(raw, mtime=0)


def bd_to_nifti_bytes(bd) -> bytes:
    """Serialize a BrainData (3D or 4D) to gzip-compressed NIfTI bytes.

    The image is sent to niivue **once** as a single volume — niivue scrubs
    4D frames natively, so there is no per-frame re-render. niivue infers the
    format from the ``.nii.gz`` name the frontend assigns, so the bytes are
    gzip-compressed to match (see `gzip_nifti`).

    Args:
        bd (BrainData): A single map (3D) or a stack (4D).

    Returns:
        The image encoded as gzip-compressed NIfTI-1 bytes.
    """
    return gzip_nifti(bd.to_nifti().to_bytes())


# --------------------------------------------------------------------------- #
# Display window (autoscaling)
# --------------------------------------------------------------------------- #

# Autoscale ceiling percentile over |finite nonzero| — a couple of outlier
# voxels must not set the whole color scale. 98 is the upper edge of the
# "robust range" convention: FSL's `fslstats -r` (2%/98% of a 1000-bin
# histogram) and niivue's own calMinMax ("robust range (2%..98%)",
# percentileFrac = 0.02). Both take the signed 2%/98%; we take the ceiling
# only, over magnitudes, because this window is symmetric about zero.
_AUTOSCALE_CEILING_PCT = 98.0
# Epsilon floor as a fraction of the ceiling: visually zero (everything above
# true zero stays visible) while zeros still render transparent. A floor, not
# a threshold — the robust range's 2% low edge would hide real voxels.
# (nilearn's plot_stat_map/view_img default to threshold=1e-6 for the same
# reason.)
_AUTOSCALE_FLOOR_FRAC = 1e-6


@dataclass(frozen=True)
class DisplayWindow:
    """The resolved niivue display window and its threshold-slider bounds.

    Both halves come out of one pass over the data so the slider handles and
    the rendered window can never disagree.

    Attributes:
        cal_min (float): Positive-limb threshold (window floor).
        cal_max (float): Positive-limb saturation endpoint (window ceiling).
        cal_min_neg (float): Negative-limb saturation endpoint.
        cal_max_neg (float): Negative-limb threshold endpoint.
        mirror_negative (bool): Keep the negative endpoints mirrored when the
            controls move.
        slider_min (float): Slider lower bound.
        slider_max (float): Slider upper bound.
        slider_value_low (float): Initial position of the low handle.
        slider_value_high (float): Initial position of the high handle.
        slider_step (float): Slider step size.
    """

    cal_min: float
    cal_max: float
    cal_min_neg: float
    cal_max_neg: float
    mirror_negative: bool
    slider_min: float
    slider_max: float
    slider_value_low: float
    slider_value_high: float
    slider_step: float


def compute_display_window(
    data,
    *,
    autoscale: bool = True,
    threshold=None,
    lower=None,
    upper=None,
    symmetric: bool | Literal["auto"] = "auto",
) -> DisplayWindow:
    """Resolve the viewer's display window and threshold-slider bounds.

    The window is always computed here and passed to niivue explicitly, so
    the slider handles can never show one window while niivue renders
    another. Precedence: ``lower``/``upper`` win; otherwise ``threshold``
    sets the floor; any **unset** edge comes from ``autoscale``:

    - ``True`` (default): ceiling = 98th percentile of the finite nonzero
      magnitudes (robust to outliers), floor = an epsilon just above zero,
      never above the smallest nonzero magnitude (zeros render transparent,
      every real voxel shows — threshold up from there).
    - ``False``: the raw magnitude range from zero to the largest absolute value.

    For a custom percentile window, pass ``lower``/``upper`` as percentile
    strings (``lower="60%", upper="98%"``) rather than a second spelling of
    the same thing on ``autoscale``.

    ``threshold`` / ``lower`` / ``upper`` accept percentile strings
    (``"98%"``), resolved over the finite nonzero **magnitudes** via
    `nltools.utils.resolve_threshold` — the viewer's window is a divergent
    magnitude window, so its percentiles are magnitude percentiles.

    ``symmetric='auto'`` mirrors the positive and negative limbs only for
    mixed-signed data. ``False`` scales each present sign independently;
    ``True`` always mirrors.

    The slider spans the data's finite value range, widened as needed to
    include the resolved window so it is always representable (never silently
    clamped), with its handles at the window edges.

    Args:
        data (np.ndarray): The BrainData's data array.
        autoscale (bool): See above.
        threshold (float | str | None): Symmetric magnitude floor (ignored when
            ``lower``/``upper`` are given).
        lower (float | str | None): Explicit window floor.
        upper (float | str | None): Explicit window ceiling.
        symmetric (bool | str): ``True``, ``False``, or ``'auto'``. See above.

    Returns:
        DisplayWindow: The window endpoints and the slider bounds.

    Raises:
        TypeError: If ``autoscale`` is not a bool, or ``symmetric`` is not
            ``True``, ``False``, or ``'auto'``.
    """
    import numpy as np

    from nltools.utils import resolve_threshold

    if not isinstance(autoscale, bool):
        raise TypeError("autoscale must be a bool")
    if not (isinstance(symmetric, bool) or symmetric == "auto"):
        raise TypeError("symmetric must be True, False, or 'auto'")

    # One pass over the array feeds every population below: the magnitudes the
    # percentile specs resolve against, the per-sign limbs, and the slider's
    # finite range.
    arr = np.asarray(data, dtype=float)
    finite = arr[np.isfinite(arr)]
    nonzero = finite[finite != 0]
    magnitudes = np.abs(nonzero)
    positive = nonzero[nonzero > 0]
    negative_magnitudes = np.abs(nonzero[nonzero < 0])

    def _mag_pct(pct: float) -> float:
        if magnitudes.size == 0:
            return 0.0
        return float(np.percentile(magnitudes, pct))

    # Resolve percentile strings against the magnitude distribution.
    # `magnitudes` (finite nonzero |values|) is exactly the population
    # resolve_threshold would keep after its own finite/nonzero filtering, so
    # reuse it instead of materializing np.abs(arr) three times per call
    # (~540 MB of transients on a 100×228k float64 map).
    threshold = resolve_threshold(threshold, magnitudes)
    lower = resolve_threshold(lower, magnitudes)
    upper = resolve_threshold(upper, magnitudes)

    # Precedence: lower/upper win; else threshold sets the floor.
    if lower is not None or upper is not None:
        floor, ceiling = lower, upper
    elif threshold is not None:
        floor, ceiling = threshold, None
    else:
        floor, ceiling = None, None

    if autoscale is False:
        ceiling = (
            float(magnitudes.max()) if ceiling is None and magnitudes.size else ceiling
        )
        ceiling = 1.0 if ceiling is None else float(ceiling)
        floor = 0.0 if floor is None else float(floor)
    else:
        if ceiling is None:
            ceiling = _mag_pct(_AUTOSCALE_CEILING_PCT)
            if ceiling == 0.0:
                ceiling = 1.0  # empty / all-zero map: keep a sane window
        if floor is None:
            # The default floor exists to make stored zeros transparent, not to
            # threshold. A bare fraction of the ceiling would start hiding real
            # voxels once the map's dynamic range exceeds 1 / the fraction, so
            # clamp it to the smallest nonzero magnitude.
            epsilon = ceiling * _AUTOSCALE_FLOOR_FRAC
            floor = (
                min(epsilon, float(magnitudes.min())) if magnitudes.size else epsilon
            )
        floor, ceiling = float(floor), float(ceiling)

    cal_min = floor
    use_symmetric = symmetric is True or (
        symmetric == "auto" and positive.size > 0 and negative_magnitudes.size > 0
    )

    if use_symmetric:
        cal_max, cal_min_neg, cal_max_neg = ceiling, -ceiling, -cal_min
    else:
        explicit_ceiling = upper is not None

        def _ceiling(sign_values, fallback):
            if explicit_ceiling or sign_values.size == 0:
                return float(fallback)
            if autoscale:
                return float(np.percentile(sign_values, _AUTOSCALE_CEILING_PCT))
            return float(sign_values.max())

        cal_max = _ceiling(positive, ceiling)
        cal_min_neg = -_ceiling(negative_magnitudes, ceiling)
        cal_max_neg = -cal_min

    if finite.size == 0:
        slider_min, slider_max = 0.0, 1.0
    else:
        slider_min, slider_max = float(finite.min()), float(finite.max())

    # Widen the range to include the resolved window so its handles land
    # exactly where asked rather than being clamped to the data extremes.
    slider_min = min(slider_min, cal_min, cal_max)
    slider_max = max(slider_max, cal_min, cal_max)
    if slider_min == slider_max:
        slider_max = slider_min + 1.0

    return DisplayWindow(
        cal_min=cal_min,
        cal_max=cal_max,
        cal_min_neg=cal_min_neg,
        cal_max_neg=cal_max_neg,
        mirror_negative=use_symmetric,
        slider_min=slider_min,
        slider_max=slider_max,
        slider_value_low=cal_min,
        slider_value_high=cal_max,
        slider_step=(slider_max - slider_min) / 200.0 or 0.01,
    )


# --------------------------------------------------------------------------- #
# Widget (imperative shell)
# --------------------------------------------------------------------------- #


class NiivueViewer(anywidget.AnyWidget):
    """anywidget wrapper around ``@niivue/niivue``, driven via the standard API.

    Holds the volume stack as byte traits (``bg_bytes`` / ``statmap_bytes`` /
    ``atlas_bytes``, any empty and skipped) plus display-parameter traits that
    ``viewer.js`` reads to configure niivue. Scalar traits (``cal_min`` /
    ``cal_max`` / negative endpoints / ``slice_type`` / ``colorbar`` /
    ``atlas_outline``) are
    reactive: set them from Python and the frontend updates in place; the
    in-widget threshold slider writes ``cal_min`` / ``cal_max`` back.

    Not constructed directly — `build_viewer` fills it from a BrainData.
    """

    _esm = _VIEWER_JS

    # Volume bytes (empty == absent). NIfTI-1, sent as one buffer each.
    bg_bytes = traitlets.Bytes(b"").tag(sync=True)
    statmap_bytes = traitlets.Bytes(b"").tag(sync=True)
    atlas_bytes = traitlets.Bytes(b"").tag(sync=True)

    # Display params for the stat map (name/colormap/colormap_negative/opacity)
    # and the atlas overlay (name + integer-indexed label LUT).
    statmap = traitlets.Dict().tag(sync=True)
    atlas_name = traitlets.Unicode("").tag(sync=True)
    atlas_lut = traitlets.Dict().tag(sync=True)

    # Reactive stat-map window; None == niivue auto (percentile-derived).
    cal_min = traitlets.Float(None, allow_none=True).tag(sync=True)
    cal_max = traitlets.Float(None, allow_none=True).tag(sync=True)
    cal_min_neg = traitlets.Float(None, allow_none=True).tag(sync=True)
    cal_max_neg = traitlets.Float(None, allow_none=True).tag(sync=True)
    mirror_negative = traitlets.Bool(True).tag(sync=True)

    slice_type = traitlets.Unicode("MULTIPLANAR").tag(sync=True)
    colorbar = traitlets.Bool(True).tag(sync=True)
    atlas_outline = traitlets.Float(0.0).tag(sync=True)

    # Controls + layout.
    controls = traitlets.Bool(True).tag(sync=True)
    slider_bounds = traitlets.Dict().tag(sync=True)
    height = traitlets.Int(600).tag(sync=True)

    # Extra niivue ConfigOptions forwarded verbatim to ``new Niivue(opts)``.
    niivue_opts = traitlets.Dict().tag(sync=True)


def build_viewer(
    bd,
    *,
    window: DisplayWindow,
    view: str = "ortho",
    cmap: str | None = None,
    atlas: str | Atlas | None = None,
    bg_img: str | bool | None = None,
    opacity: float = 1.0,
    outline: float = 0.0,
    colorbar: bool = True,
    controls: bool = True,
    niivue_opts: dict | None = None,
) -> NiivueViewer:
    """Assemble a configured `NiivueViewer` for a BrainData.

    Builds the volume stack ``[background?, statmap, atlas?]`` (atlas on top
    so its outlines/opacity keep the stat map readable) as byte + parameter
    traits, and sets the slice type.

    Args:
        bd (BrainData): BrainData to view.
        window: The resolved display window and slider bounds. Required, and
            must be built from ``bd.data`` with `compute_display_window` — the
            slider handles and the rendered window come from it together, so
            they can never disagree.
        view: See `slice_type_for`.
        cmap: Positive colormap (niivue or matplotlib name). ``None`` uses the
            sign-aware red-positive/blue-negative default.
        atlas: Atlas name, `Atlas`, or ``None``.
        bg_img: See `resolve_background`.
        opacity: Stat-map (and filled-atlas) opacity.
        outline: ``> 0`` draws atlas region boundaries of that width;
            ``0`` draws filled regions.
        colorbar: Show the stat-map colorbar (the ``cmap`` scale). Only the
            stat map carries a colorbar; the background and atlas overlays are
            suppressed by the frontend.
        controls: Render the in-widget threshold slider (default ``True``).
        niivue_opts: Extra kwargs forwarded verbatim to ``new Niivue(opts)``.
            A ``height`` key overrides the 600px ortho and 400px single-view
            defaults; an ``is_colorbar`` key overrides ``colorbar``.

    Returns:
        NiivueViewer: A configured widget ready to display.
    """
    # niivue selects the positive/negative limb from the voxel sign, so this
    # pair is already sign-aware without swapping colormap names per map.
    cmap_resolved = "warm" if cmap is None else resolve_cmap(cmap)
    cmap_negative = divergent_partner(cmap_resolved)
    slice_name = slice_type_for(view)
    atlas_obj = _coerce_atlas(atlas)
    bg_path = resolve_background(bd.mask.affine, bg_img)

    # Validate / compute the atlas LUT up front so a probabilistic atlas
    # raises before we serialize any image bytes.
    atlas_lut = atlas_to_label_lut(atlas_obj) if atlas_obj is not None else {}

    # Pull height / is_colorbar out of the forwarded niivue opts: height is a
    # canvas-layout trait, and an explicit is_colorbar wins over colorbar=.
    opts = dict(niivue_opts or {})
    height = int(opts.pop("height", 600 if view == "ortho" else 400))
    if "is_colorbar" in opts:
        colorbar = bool(opts.pop("is_colorbar"))

    return NiivueViewer(
        bg_bytes=gzip_nifti(pathlib.Path(bg_path).read_bytes()) if bg_path else b"",
        statmap_bytes=bd_to_nifti_bytes(bd),
        atlas_bytes=(
            gzip_nifti(atlas_obj.image.to_bytes()) if atlas_obj is not None else b""
        ),
        statmap={
            "name": "statmap",
            "colormap": cmap_resolved,
            "colormap_negative": cmap_negative,
            "opacity": float(opacity),
        },
        atlas_name=atlas_obj.name if atlas_obj is not None else "",
        atlas_lut=atlas_lut,
        cal_min=window.cal_min,
        cal_max=window.cal_max,
        cal_min_neg=window.cal_min_neg,
        cal_max_neg=window.cal_max_neg,
        mirror_negative=window.mirror_negative,
        slice_type=slice_name,
        colorbar=bool(colorbar),
        atlas_outline=float(outline) if atlas_obj is not None else 0.0,
        controls=bool(controls),
        slider_bounds={
            "min": window.slider_min,
            "max": window.slider_max,
            "value_low": window.slider_value_low,
            "value_high": window.slider_value_high,
            "step": window.slider_step,
        },
        height=height,
        niivue_opts=opts,
    )
