"""Tests for nltools.templates — brain-space configuration and helpers."""

import os

import nibabel as nib
import numpy as np
import pytest

from nltools.templates import (
    BrainSpaceConfig,
    TemplateMatch,
    get_bg_image,
    get_brainspace,
    match_resolution,
    reset_brainspace,
    resolve_paths,
    resolve_template_name,
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

    def test_set_resolution_only(self):
        new = set_brainspace(resolution=3)
        assert new.template == "default"
        assert new.resolution == 3

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
        with pytest.raises(RuntimeError):
            with with_brainspace(template="fmriprep", resolution=1):
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
# resolve_paths
# ---------------------------------------------------------------------------


class TestResolvePaths:
    @pytest.mark.parametrize(
        "template,resolution",
        [
            ("default", 2),
            ("default", 3),
            ("nilearn", 1),
            ("nilearn", 2),
            ("nilearn", 3),
            ("fmriprep", 1),
            ("fmriprep", 2),
        ],
    )
    def test_all_valid_combinations(self, template, resolution):
        paths = resolve_paths(template, resolution)
        for key in ("mask", "brain", "plot"):
            assert os.path.exists(paths[key]), f"missing: {paths[key]}"
            assert f"{resolution}mm" in paths[key]

    def test_unknown_template(self):
        with pytest.raises(ValueError, match="Unknown template"):
            resolve_paths("bogus", 2)

    def test_unsupported_resolution(self):
        with pytest.raises(ValueError, match="not supported"):
            resolve_paths("default", 1)


# ---------------------------------------------------------------------------
# resolve_template_name
# ---------------------------------------------------------------------------


class TestResolveTemplateName:
    @pytest.mark.parametrize(
        "name,expect_substr",
        [
            ("2mm-MNI152-2009c", "fmriprep"),
            ("3mm-MNI152-2009a", "nilearn"),
            ("2mm-MNI152-2009fsl", "default"),
            ("1mm-MNI152-2009c", "fmriprep"),
        ],
    )
    def test_valid_names(self, name, expect_substr):
        path = resolve_template_name(name, file_type="mask")
        assert os.path.exists(path)
        assert expect_substr in path

    def test_default_file_type_is_mask(self):
        p = resolve_template_name("2mm-MNI152-2009c")
        assert "mask" in os.path.basename(p)

    def test_brain_file_type(self):
        p = resolve_template_name("2mm-MNI152-2009c", file_type="brain")
        assert "brain" in os.path.basename(p)

    def test_t1_file_type(self):
        p = resolve_template_name("2mm-MNI152-2009c", file_type="T1")
        assert "T1" in os.path.basename(p)

    def test_invalid_file_type(self):
        with pytest.raises(ValueError, match="file_type"):
            resolve_template_name("2mm-MNI152-2009c", file_type="bogus")

    def test_invalid_name_format(self):
        with pytest.raises(ValueError, match="Invalid template name format"):
            resolve_template_name("not-a-template")

    def test_unknown_version_code(self):
        # "f" matches the format regex ([acfsl]+) but isn't a valid version
        with pytest.raises(ValueError, match="Unknown version code"):
            resolve_template_name("2mm-MNI152-2009f")


# ---------------------------------------------------------------------------
# match_resolution
# ---------------------------------------------------------------------------


def _isotropic_affine(mm: float) -> np.ndarray:
    aff = np.eye(4)
    aff[0, 0] = mm
    aff[1, 1] = mm
    aff[2, 2] = mm
    return aff


class TestMatchResolution:
    def test_exact_2mm(self):
        m = match_resolution(_isotropic_affine(2.0))
        assert isinstance(m, TemplateMatch)
        assert m.resolution == 2
        assert m.match_distance == 0
        assert os.path.exists(m.mask_path)

    def test_exact_1mm(self):
        m = match_resolution(_isotropic_affine(1.0))
        assert m.resolution == 1
        # default doesn't support 1mm, so nilearn wins (comes before fmriprep)
        assert m.template == "nilearn"

    def test_exact_3mm(self):
        m = match_resolution(_isotropic_affine(3.0))
        assert m.resolution == 3
        assert m.template == "default"

    def test_non_isotropic_uses_mean(self):
        aff = np.eye(4)
        aff[0, 0] = 2.0
        aff[1, 1] = 2.0
        aff[2, 2] = 2.5
        m = match_resolution(aff)
        # mean ≈ 2.17 rounds to 2
        assert m.resolution == 2

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError, match="outside"):
            match_resolution(_isotropic_affine(15.0))

    def test_warns_on_resample(self, recwarn):
        # 4mm isn't supported exactly; falls back to closest (3)
        match_resolution(_isotropic_affine(4.0), warn_resample=True)
        assert any("doesn't exactly match" in str(w.message) for w in recwarn.list)

    def test_no_warning_when_disabled(self, recwarn):
        match_resolution(_isotropic_affine(4.0), warn_resample=False)
        assert not any("doesn't exactly match" in str(w.message) for w in recwarn.list)


# ---------------------------------------------------------------------------
# get_bg_image
# ---------------------------------------------------------------------------


class TestGetBgImage:
    def test_matching_resolution_returns_template_path(self):
        mask = nib.load(get_brainspace().mask)
        path = get_bg_image(mask.affine, img_type="brain")
        assert os.path.exists(path)
        assert "brain" in os.path.basename(path)

    def test_plot_type(self):
        mask = nib.load(get_brainspace().mask)
        path = get_bg_image(mask.affine, img_type="plot")
        assert "T1" in os.path.basename(path)

    def test_invalid_img_type(self):
        mask = nib.load(get_brainspace().mask)
        with pytest.raises(ValueError, match="img_type"):
            get_bg_image(mask.affine, img_type="bogus")

    def test_non_isotropic_raises(self):
        aff = np.eye(4)
        aff[0, 0] = 2.0
        aff[1, 1] = 2.0
        aff[2, 2] = 3.0
        with pytest.raises(ValueError, match="isotropic"):
            get_bg_image(aff)

    def test_unsupported_resolution_falls_back_to_config(self):
        # default template doesn't support 1mm; should fall back to cfg.brain
        set_brainspace(template="default", resolution=2)
        path = get_bg_image(_isotropic_affine(1.0), img_type="brain")
        assert path == get_brainspace().brain

    def test_explicit_config_argument(self):
        cfg = BrainSpaceConfig(template="fmriprep", resolution=2)
        path = get_bg_image(_isotropic_affine(2.0), img_type="brain", config=cfg)
        assert "fmriprep" in path
