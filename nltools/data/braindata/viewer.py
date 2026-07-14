"""ipyniivue (niivue) interactive viewer for BrainData.

`build_viewer` returns a configured `NiiVue` — a
WebGL brain viewer with live windowing (right-drag), slice scrolling,
native 4D frame scrubbing, true 3D rendering, and optional nltools-atlas
overlays (colored regions / outlines / hover labels). Live-kernel only
(Jupyter, marimo); static docs keep using `BrainData.plot`.

The module is split functional-core / imperative-shell:

- Pure helpers (`resolve_cmap`, `divergent_partner`,
  `slice_type_for`, `qualitative_colors`,
  `atlas_to_label_lut`, `resolve_background`,
  `bd_to_volume`) translate BrainData / `Atlas` state into the
  vocabulary niivue understands.
- `build_viewer` is the thin assembler that constructs the widget,
  loads the volume stack, and applies the display settings.

niivue formatting deliberately lives here, not in
``nltools/data/atlases/`` — the atlas package stays niivue-agnostic and
only exposes the generic `Atlas` dataclass.
"""

from __future__ import annotations

import colorsys
import functools
import pathlib
import warnings

from nltools.data.atlases import Atlas, load_atlas
from nltools.templates.matching import get_bg_image, is_standard_space

try:
    from ipyniivue import NiiVue, SliceType, Volume
except ModuleNotFoundError as exc:  # pragma: no cover - hard dep, friendly hint
    raise ModuleNotFoundError(
        "BrainData.iplot() requires the optional viewer backend `ipyniivue`. "
        "Install it with `uv add ipyniivue` (or `pip install ipyniivue`)."
    ) from exc


# --------------------------------------------------------------------------- #
# Colormaps
# --------------------------------------------------------------------------- #

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
    """Names of niivue's builtin colormaps (read from the package's statics).

    Cached; avoids constructing a throwaway ``NiiVue`` widget just to call
    ``nv.colormaps()``.
    """
    import ipyniivue

    cmap_dir = pathlib.Path(ipyniivue.__file__).parent / "static" / "colormaps"
    return frozenset(
        p.stem.lower() for p in cmap_dir.glob("*.json") if not p.stem.startswith("$")
    )


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
            stacklevel=2,
        )
        return mapped
    warnings.warn(
        f"colormap {name!r} is not a known niivue colormap; falling back to 'warm'.",
        stacklevel=2,
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

_VIEW_TO_SLICE: dict[str, SliceType] = {
    "ortho": SliceType.MULTIPLANAR,
    "axial": SliceType.AXIAL,
    "coronal": SliceType.CORONAL,
    "sagittal": SliceType.SAGITTAL,
    "render": SliceType.RENDER,
}


def slice_type_for(view: str) -> SliceType:
    """Map a ``view`` string to a niivue `SliceType`.

    Args:
        view: One of ``"ortho"``, ``"axial"``, ``"coronal"``,
            ``"sagittal"``, ``"render"``.

    Returns:
        The matching `SliceType`.

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
        ``{"R", "G", "B", "A", "labels"}`` dict suitable for
        `set_colormap_label`.

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
        affine: 4x4 affine of the BrainData (``bd.mask.affine``), used to
            decide whether auto-MNI applies.
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
# Volumes
# --------------------------------------------------------------------------- #


def bd_to_volume(
    bd,
    *,
    name: str,
    cmap: str,
    cmap_negative: str,
    cal_min: float | None,
    cal_max: float | None,
    opacity: float,
) -> Volume:
    """Build a niivue `Volume` from a BrainData stat map.

    The 3D (``bd[0]``) or 4D (a stack) image is loaded **once** as a single
    volume — niivue scrubs 4D frames natively, so there is no per-frame
    re-render. Thresholding is the divergent magnitude window
    ``[cal_min, cal_max]``: the positive side uses ``cmap``, the negative
    side mirrors it via ``cmap_negative``. ``cal_min`` is the display floor;
    because niivue's overlay colormaps ramp alpha to 0 at the floor,
    sub-floor voxels render transparent (true thresholding).

    Args:
        bd: A BrainData (3D for a single map, 4D for a stack).
        name: Volume name (shown in the colorbar / legend).
        cmap: Resolved niivue positive colormap.
        cmap_negative: niivue colormap for negative values.
        cal_min: Window floor, or ``None`` for niivue auto.
        cal_max: Window ceiling, or ``None`` for niivue auto.
        opacity: Volume opacity in ``0..1``.

    Returns:
        A configured `Volume`.
    """
    vol = Volume(
        name=name,
        data=bd.to_nifti().to_bytes(),
        colormap=cmap,
        colormap_negative=cmap_negative,
        opacity=float(opacity),
    )
    if cal_min is not None:
        vol.cal_min = float(cal_min)
    if cal_max is not None:
        vol.cal_max = float(cal_max)
    return vol


# --------------------------------------------------------------------------- #
# Assembler (imperative shell)
# --------------------------------------------------------------------------- #


def build_viewer(
    bd,
    *,
    view: str = "ortho",
    cal_min: float | None = None,
    cal_max: float | None = None,
    cmap: str = "warm",
    atlas: str | Atlas | None = None,
    bg_img: str | bool | None = None,
    opacity: float = 1.0,
    outline: float = 0.0,
    niivue_opts: dict | None = None,
) -> NiiVue:
    """Assemble a configured `NiiVue` for a BrainData.

    Builds the volume stack ``[background?, statmap, atlas?]`` (atlas on top
    so its outlines/opacity keep the stat map readable), loads it, applies
    the atlas label LUT / outline, and sets the slice type.

    Args:
        bd: BrainData to view.
        view: See `slice_type_for`.
        cal_min: Window floor (threshold), or ``None`` for auto.
        cal_max: Window ceiling, or ``None`` for auto.
        cmap: Positive colormap (niivue or matplotlib name).
        atlas: Atlas name, `Atlas`, or ``None``.
        bg_img: See `resolve_background`.
        opacity: Stat-map (and filled-atlas) opacity.
        outline: ``> 0`` draws atlas region boundaries of that width;
            ``0`` draws filled regions.
        niivue_opts: Extra kwargs forwarded verbatim to ``NiiVue(**opts)``.

    Returns:
        A configured `NiiVue` ready to display.
    """
    cmap_resolved = resolve_cmap(cmap)
    cmap_negative = divergent_partner(cmap_resolved)
    slice_type = slice_type_for(view)
    atlas_obj = _coerce_atlas(atlas)
    bg_path = resolve_background(bd.mask.affine, bg_img)

    # Validate / compute the atlas LUT up front so a probabilistic atlas
    # raises before we serialize any image bytes.
    atlas_lut = atlas_to_label_lut(atlas_obj) if atlas_obj is not None else None

    statmap = bd_to_volume(
        bd,
        name="statmap",
        cmap=cmap_resolved,
        cmap_negative=cmap_negative,
        cal_min=cal_min,
        cal_max=cal_max,
        opacity=opacity,
    )

    volumes: list[Volume] = []
    if bg_path is not None:
        volumes.append(Volume(path=bg_path, name="background", colormap="gray"))
    volumes.append(statmap)
    if atlas_obj is not None:
        atlas_vol = Volume(
            name=atlas_obj.name,
            data=atlas_obj.image.to_bytes(),
            opacity=float(opacity),
        )
        atlas_vol.set_colormap_label(atlas_lut)
        volumes.append(atlas_vol)

    nv = NiiVue(**(niivue_opts or {}))
    nv.load_volumes(volumes)
    if atlas_obj is not None and outline > 0:
        nv.set_atlas_outline(float(outline))
    nv.set_slice_type(slice_type)
    return nv
