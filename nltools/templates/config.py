"""Global brain-space configuration: frozen dataclass + set/get/with API."""

from contextlib import contextmanager
from dataclasses import dataclass, replace
from collections.abc import Iterator

from .paths import resolve_paths
from .registry import SUPPORTED_RESOLUTIONS, TemplateName, Resolution


@dataclass(frozen=True)
class BrainSpaceConfig:
    """Immutable MNI template configuration.

    Attributes:
        template: Template variant (``'default'``, ``'nilearn'``, ``'fmriprep'``).
        resolution: Resolution in mm (1, 2, or 3).
    """

    template: TemplateName = "default"
    resolution: Resolution = 2

    def __post_init__(self) -> None:
        if self.template not in SUPPORTED_RESOLUTIONS:
            raise ValueError(
                f"Unknown template: {self.template!r}. "
                f"Supported: {sorted(SUPPORTED_RESOLUTIONS)}"
            )
        if self.resolution not in SUPPORTED_RESOLUTIONS[self.template]:
            raise ValueError(
                f"Resolution {self.resolution}mm not supported for "
                f"{self.template!r}. "
                f"Supported: {SUPPORTED_RESOLUTIONS[self.template]}"
            )

    @property
    def mask(self) -> str:
        """Path to the brain mask file."""
        return resolve_paths(self.template, self.resolution)["mask"]

    @property
    def brain(self) -> str:
        """Path to the brain-extracted image."""
        return resolve_paths(self.template, self.resolution)["brain"]

    @property
    def plot(self) -> str:
        """Path to the full T1 image used for plotting."""
        return resolve_paths(self.template, self.resolution)["plot"]

    def __repr__(self) -> str:
        import os

        return (
            f"BrainSpaceConfig(template={self.template!r}, "
            f"resolution={self.resolution}mm)\n"
            f"  mask: {os.path.basename(self.mask)}\n"
            f"  brain: {os.path.basename(self.brain)}\n"
            f"  plot: {os.path.basename(self.plot)}"
        )


_DEFAULT = BrainSpaceConfig()
_current: BrainSpaceConfig = _DEFAULT


def get_brainspace() -> BrainSpaceConfig:
    """Return the current global brain-space configuration."""
    return _current


def set_brainspace(
    template: TemplateName | None = None,
    resolution: Resolution | None = None,
) -> BrainSpaceConfig:
    """Set the global brain-space configuration.

    Call with no arguments to return the current config without mutating it.
    Call with one or both arguments to mutate the global state; unspecified
    fields retain their current value.

    Args:
        template: Template name to set. If ``None``, keeps current.
        resolution: Resolution to set. If ``None``, keeps current.

    Returns:
        The new (or unchanged) current ``BrainSpaceConfig``.
    """
    global _current
    if template is None and resolution is None:
        return _current
    updates: dict[str, object] = {}
    if template is not None:
        updates["template"] = template
    if resolution is not None:
        updates["resolution"] = resolution
    _current = replace(_current, **updates)
    return _current


def reset_brainspace() -> BrainSpaceConfig:
    """Reset the global brain-space configuration to defaults."""
    global _current
    _current = _DEFAULT
    return _current


@contextmanager
def with_brainspace(
    template: TemplateName | None = None,
    resolution: Resolution | None = None,
) -> Iterator[BrainSpaceConfig]:
    """Temporarily change the global brain-space configuration.

    Restores the previous configuration on exit, even if an exception is
    raised inside the block.

    Args:
        template: Template name for the duration of the block.
        resolution: Resolution for the duration of the block.

    Yields:
        The ``BrainSpaceConfig`` active inside the block.
    """
    global _current
    previous = _current
    try:
        set_brainspace(template=template, resolution=resolution)
        yield _current
    finally:
        _current = previous
