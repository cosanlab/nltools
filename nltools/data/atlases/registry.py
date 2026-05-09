"""Static registry of atlases hosted at ``nltools/niftis/atlases``.

Each entry describes an atlas's kind (deterministic vs probabilistic) and
the citation users should cite when they use it. The actual NIfTI + label
files are fetched lazily by :func:`nltools.data.atlases.load_atlas` via
:func:`nltools.templates.fetch_resource`.

Atlases were sourced from atlasreader (BSD-3-Clause) and are subject to
their original upstream licenses — see ``LICENSES.md`` in the HF dataset.
"""

from dataclasses import dataclass
from typing import Literal

AtlasKind = Literal["deterministic", "probabilistic"]


@dataclass(frozen=True)
class AtlasMetadata:
    """Static description of a registered atlas.

    Attributes:
        kind: ``"deterministic"`` (3D integer-labeled) or
            ``"probabilistic"`` (4D, last axis indexes regions).
        citation: Short citation string for the original atlas.
    """

    kind: AtlasKind
    citation: str


ATLASES: dict[str, AtlasMetadata] = {
    "aal": AtlasMetadata(
        kind="deterministic",
        citation="Tzourio-Mazoyer et al. 2002, NeuroImage",
    ),
    "aicha": AtlasMetadata(
        kind="deterministic",
        citation="Joliot et al. 2015, J Neurosci Methods",
    ),
    "desikan_killiany": AtlasMetadata(
        kind="deterministic",
        citation="Desikan et al. 2006, NeuroImage (FreeSurfer license)",
    ),
    "destrieux": AtlasMetadata(
        kind="deterministic",
        citation="Destrieux et al. 2010, NeuroImage (FreeSurfer license)",
    ),
    "harvard_oxford": AtlasMetadata(
        kind="probabilistic",
        citation="Desikan et al. 2006, NeuroImage / FSL Harvard-Oxford",
    ),
    "juelich": AtlasMetadata(
        kind="probabilistic",
        citation="Eickhoff et al. 2005, NeuroImage",
    ),
    "marsatlas": AtlasMetadata(
        kind="deterministic",
        citation="Auzias et al. 2016, Hum Brain Mapp",
    ),
    "neuromorphometrics": AtlasMetadata(
        kind="deterministic",
        citation="MICCAI 2012 Multi-Atlas Labeling Challenge",
    ),
    "schaefer_200": AtlasMetadata(
        kind="deterministic",
        citation="Schaefer et al. 2018, Cereb Cortex (200-parcel, 7-network)",
    ),
    "talairach_ba": AtlasMetadata(
        kind="deterministic",
        citation="Talairach & Tournoux 1988 (Brodmann areas)",
    ),
    "talairach_gyrus": AtlasMetadata(
        kind="deterministic",
        citation="Talairach & Tournoux 1988 (gyri)",
    ),
}


# Trio used as the default for ``BrainData.cluster_report`` and
# ``label_coords`` — picked to give one probabilistic, one anatomical
# deterministic, and one functional deterministic atlas at once.
DEFAULT_ATLASES: tuple[str, ...] = ("harvard_oxford", "aal", "schaefer_200")


def list_atlases() -> list[str]:
    """Return the sorted list of registered atlas names.

    Returns:
        Sorted list of atlas names usable with
        :func:`nltools.data.atlases.load_atlas`.
    """
    return sorted(ATLASES.keys())
