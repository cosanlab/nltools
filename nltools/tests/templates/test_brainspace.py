"""Tests for nltools.templates — brain-space configuration and helpers."""

import os

import nibabel as nib
import numpy as np
import pytest

from nltools.templates import (
    BrainSpaceConfig,
    _get_bg_image,
    get_brainspace,
    reset_brainspace,
    _resolve_template_name,
    set_brainspace,
    with_brainspace,
)


@pytest.fixture(autouse=True)
def _reset_global():
    """Every test starts and ends with the default global config."""
    reset_brainspace()
    yield
    reset_brainspace()


# ---------------------------------------------------------------------------
# BrainSpaceConfig
# ---------------------------------------------------------------------------


class TestBrainSpaceConfig:
    def test_defaults(self):
        cfg = BrainSpaceConfig()
        assert cfg.template == "default"
        assert cfg.resolution == 2

    def test_is_frozen(self):
        cfg = BrainSpaceConfig()
        with pytest.raises(Exception):
            cfg.template = "fmriprep"  # type: ignore[misc]

    def test_invalid_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            BrainSpaceConfig(template="bogus")  # type: ignore[arg-type]

    def test_invalid_resolution_raises(self):
        with pytest.raises(ValueError, match="not supported"):
            BrainSpaceConfig(template="default", resolution=1)  # type: ignore[arg-type]

    def test_path_properties_resolve(self):
        cfg = BrainSpaceConfig(template="default", resolution=2)
        assert os.path.exists(cfg.mask)
        assert os.path.exists(cfg.brain)
        assert os.path.exists(cfg.plot)
        assert "2mm" in cfg.mask
        assert "default" in cfg.mask

    def test_repr_includes_template_and_resolution(self):
        cfg = BrainSpaceConfig(template="fmriprep", resolution=2)
        r = repr(cfg)
        assert "fmriprep" in r
        assert "2mm" in r


# ---------------------------------------------------------------------------
# set / get / reset
# ---------------------------------------------------------------------------


class TestSetGet:
    def test_get_returns_default(self):
        cfg = get_brainspace()
        assert cfg.template == "default"
        assert cfg.resolution == 2

    def test_set_no_args_returns_current_without_mutating(self):
        before = get_brainspace()
        returned = set_brainspace()
        assert returned is before
        assert get_brainspace() is before

    def test_set_template_only(self):
        new = set_brainspace(template="fmriprep")
        assert new.template == "fmriprep"
        assert new.resolution == 2  # preserved
        assert get_brainspace() is new

    def test_set_both(self):
        new = set_brainspace(template="nilearn", resolution=1)
        assert new.template == "nilearn"
        assert new.resolution == 1

    def test_set_invalid_combo_raises(self):
        # default only supports 2 and 3
        with pytest.raises(ValueError):
            set_brainspace(template="default", resolution=1)

    def test_reset(self):
        set_brainspace(template="fmriprep", resolution=1)
        cfg = reset_brainspace()
        assert cfg.template == "default"
        assert cfg.resolution == 2
        assert get_brainspace().template == "default"


# ---------------------------------------------------------------------------
# with_brainspace context manager
# ---------------------------------------------------------------------------


class TestWithBrainspace:
    def test_scopes_change(self):
        assert get_brainspace().template == "default"
        with with_brainspace(template="fmriprep", resolution=1) as cfg:
            assert cfg.template == "fmriprep"
            assert cfg.resolution == 1
            assert get_brainspace().template == "fmriprep"
        assert get_brainspace().template == "default"
        assert get_brainspace().resolution == 2

    def test_restores_on_exception(self):
        with (
            pytest.raises(RuntimeError),
            with_brainspace(template="fmriprep", resolution=1),
        ):
            raise RuntimeError("boom")
        assert get_brainspace().template == "default"
        assert get_brainspace().resolution == 2

    def test_nested(self):
        with with_brainspace(template="fmriprep", resolution=2):
            assert get_brainspace().template == "fmriprep"
            with with_brainspace(template="nilearn", resolution=1):
                assert get_brainspace().template == "nilearn"
                assert get_brainspace().resolution == 1
            assert get_brainspace().template == "fmriprep"
            assert get_brainspace().resolution == 2
        assert get_brainspace().template == "default"

    def test_partial_override(self):
        set_brainspace(template="nilearn", resolution=2)
        with with_brainspace(resolution=3):
            assert get_brainspace().template == "nilearn"
            assert get_brainspace().resolution == 3
        assert get_brainspace().resolution == 2


# ---------------------------------------------------------------------------
# _resolve_paths
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _resolve_template_name
# ---------------------------------------------------------------------------


class TestResolveTemplateName:
    @pytest.mark.parametrize(
        "name,expect_substr",
        [
            ("2mm-MNI152-2009fsl", "default"),
        ],
    )
    def test_valid_names(self, name, expect_substr):
        path = _resolve_template_name(name, file_type="mask")
        assert os.path.exists(path)
        assert expect_substr in path

    def test_default_file_type_is_mask(self):
        p = _resolve_template_name("2mm-MNI152-2009c")
        assert "mask" in os.path.basename(p)

    def test_invalid_file_type(self):
        with pytest.raises(ValueError, match="file_type"):
            _resolve_template_name("2mm-MNI152-2009c", file_type="bogus")

    def test_invalid_name_format(self):
        with pytest.raises(ValueError, match="Invalid template name format"):
            _resolve_template_name("not-a-template")

    def test_unknown_version_code(self):
        # "f" has the template-name shape but isn't a valid version code
        with pytest.raises(ValueError, match="Unknown version code"):
            _resolve_template_name("2mm-MNI152-2009f")


# ---------------------------------------------------------------------------
# _match_resolution
# ---------------------------------------------------------------------------


def _isotropic_affine(mm: float) -> np.ndarray:
    aff = np.eye(4)
    aff[0, 0] = mm
    aff[1, 1] = mm
    aff[2, 2] = mm
    return aff


# ---------------------------------------------------------------------------
# _get_bg_image
# ---------------------------------------------------------------------------


class TestGetBgImage:
    def test_matching_resolution_returns_template_path(self):
        mask = nib.load(get_brainspace().mask)
        path = _get_bg_image(mask.affine)
        assert os.path.exists(path)
        assert "brain" in os.path.basename(path)


# ---------------------------------------------------------------------------
# _is_standard_space
# ---------------------------------------------------------------------------


class TestIsStandardSpace:
    """Gate predicate used by plotting paths to refuse non-MNI data."""

    def test_isotropic_supported_resolutions(self):
        from nltools.templates import _is_standard_space

        for mm in (1, 2, 3):
            ok, reason = _is_standard_space(_isotropic_affine(float(mm)))
            assert ok, f"{mm}mm should be standard, got reason={reason!r}"
            assert reason is None

    def test_non_isotropic_rejected(self):
        # Miyawaki-shaped voxels: non-isotropic in subject native space.
        from nltools.templates import _is_standard_space

        aff = np.eye(4)
        aff[0, 0] = 3.3
        aff[1, 1] = 3.6
        aff[2, 2] = 6.4
        ok, reason = _is_standard_space(aff)
        assert not ok
        assert "non-isotropic" in reason


# ---------------------------------------------------------------------------
# list_resources
# ---------------------------------------------------------------------------
