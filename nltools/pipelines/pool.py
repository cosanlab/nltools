"""Pool infrastructure for multi-subject aggregation.

This module provides classes for pooling data across subjects and
enabling two-stage analyses (e.g., first-level GLM -> group t-test).

The pool() method serves as an execution boundary - everything before
it is executed lazily, and pool() triggers execution and aggregation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class PooledData:
    """Aggregated data from multiple subjects.

    PooledData serves as a checkpoint after first-level analyses,
    enabling reusable second-level analyses without re-running
    the first-level computations.

    Args:
        data: Pooled data array. Shape is (n_subjects, n_voxels) for single
            parameter or (n_subjects, n_conditions, n_voxels) for multi-condition.
        param: Parameter that was pooled (e.g., 'beta', 'residual', 't').
        condition_names: Names of conditions if multi-condition data.
        subject_ids: Subject identifiers.
        fitted_state: Saved fitted models for repool() functionality.
        save_path: Path where data was saved.

    Examples:
    >>> # Two-stage GLM
    >>> pool = bc.fit(model='glm', X=designs).pool(param='beta')
    >>> result = pool.fit(model='ttest', contrast='face-house')
    >>>
    >>> # Reuse for multiple contrasts
    >>> result1 = pool.fit(model='ttest', contrast='face-house')
    >>> result2 = pool.fit(model='ttest', contrast='face-object')
    """

    data: NDArray
    param: str
    condition_names: list[str] | None = None
    subject_ids: list[str] | None = None
    mask: Any | None = field(
        default=None, repr=False
    )  # nibabel image for reconstruction
    fitted_state: Any | None = field(default=None, repr=False)
    save_path: str | None = None

    @property
    def n_subjects(self) -> int:
        """Number of subjects in the pooled dataset (first dimension of data)."""
        return self.data.shape[0]

    @property
    def n_conditions(self) -> int | None:
        """Number of conditions (None if single-condition)."""
        if self.data.ndim == 3:
            return self.data.shape[1]
        return None

    @property
    def n_voxels(self) -> int:
        """Number of voxels (last dimension of data array)."""
        return self.data.shape[-1]

    @property
    def shape(self) -> tuple:
        """Shape of the pooled data array as (n_subjects[, n_conditions], n_voxels)."""
        return self.data.shape

    def fit(
        self,
        model: str,
        contrast: str | None = None,
        contrasts: list[str] | None = None,
        X: NDArray | None = None,
        **kwargs,
    ) -> StatResult | ResultDict:
        """Fit second-level statistical model.

        This is a terminal method - executes immediately (eager).

        Args:
            model: Statistical model type: 'ttest' (one-sample or two-sample with X),
                'paired_ttest', or 'anova'.
            contrast: Single contrast specification (e.g., 'face-house').
            contrasts: Multiple contrasts - returns ResultDict.
            X: Design matrix for two-sample tests or ANOVA.
            **kwargs: Additional arguments for the statistical test.

        Returns:
            Statistical results. ResultDict if multiple contrasts specified.

        Examples:
        >>> result = pool.fit(model='ttest', contrast='face-house')
        >>> result.t_map.max()

        >>> results = pool.fit(model='ttest', contrasts=['A-B', 'A-C', 'B-C'])
        >>> results['A-B'].threshold(method='fdr')
        """
        if contrasts is not None:
            return ResultDict(
                {c: self._fit_one(model, c, X, **kwargs) for c in contrasts}
            )
        return self._fit_one(model, contrast, X, **kwargs)

    def _fit_one(
        self,
        model: str,
        contrast: str | None,
        X: NDArray | None,
        **kwargs,
    ) -> StatResult:
        """Fit single contrast/model."""
        from scipy import stats

        # Apply contrast if specified
        if contrast is not None and self.n_conditions is not None:
            data = self._apply_contrast(contrast)
        else:
            data = self.data

        # Ensure 2D: (n_subjects, n_voxels)
        if data.ndim == 3:
            # If still 3D, take first condition
            data = data[:, 0, :]

        if model == "ttest":
            if X is None:
                # One-sample t-test against 0
                t_vals, p_vals = stats.ttest_1samp(data, 0, axis=0)
            else:
                # Two-sample based on design
                groups = np.unique(X)
                if len(groups) == 2:
                    mask1 = groups[0] == X
                    mask2 = groups[1] == X
                    t_vals, p_vals = stats.ttest_ind(
                        data[mask1.flatten()], data[mask2.flatten()], axis=0
                    )
                else:
                    raise ValueError("Two-sample t-test requires exactly 2 groups")
        elif model == "paired_ttest":
            # paired_ttest compares the two conditions directly; a contrast
            # would collapse them to a single vector, so accepting one and
            # ignoring it (returning uncontrasted results labeled with the
            # contrast) is worse than refusing it outright.
            if contrast is not None:
                raise ValueError(
                    "contrast is not supported for model='paired_ttest'; "
                    "it compares the two conditions directly."
                )
            if self.n_conditions != 2:
                raise ValueError("Paired t-test requires exactly 2 conditions")
            t_vals, p_vals = stats.ttest_rel(
                self.data[:, 0, :], self.data[:, 1, :], axis=0
            )
        elif model == "anova":
            # F-test across conditions; a contrast would collapse the
            # conditions the ANOVA is meant to test across, so refuse it
            # rather than silently ignore it.
            if contrast is not None:
                raise ValueError(
                    "contrast is not supported for model='anova'; "
                    "it tests across all conditions."
                )
            if self.n_conditions is None:
                raise ValueError("ANOVA requires multi-condition data (3D array)")
            n_cond = self.n_conditions  # Store in local variable for type checker
            f_vals, p_vals = stats.f_oneway(
                *[self.data[:, i, :] for i in range(n_cond)]
            )
            return StatResult(f_map=f_vals, p_map=p_vals, contrast=contrast)
        else:
            raise ValueError(
                f"Unknown model: {model}. Options: ttest, paired_ttest, anova"
            )

        return StatResult(t_map=t_vals, p_map=p_vals, contrast=contrast)

    def _apply_contrast(self, contrast: str) -> NDArray:
        """Apply contrast to multi-condition data.

        Args:
            contrast: Contrast specification like 'face-house' or 'A-B+C'.

        Returns:
            Contrast-weighted data, shape (n_subjects, n_voxels).
        """
        if self.condition_names is None:
            raise ValueError("Cannot apply named contrast without condition_names")

        # Parse contrast string
        weights = self._parse_contrast(contrast)

        # Apply: sum over conditions with weights
        # data: (n_subjects, n_conditions, n_voxels)
        # weights: (n_conditions,)
        result = np.tensordot(weights, self.data, axes=([0], [1]))
        return result  # (n_subjects, n_voxels)

    def _parse_contrast(self, contrast: str) -> NDArray:
        """Parse contrast string to weight vector."""
        # Caller (_apply_contrast) already checks condition_names is not None
        if self.n_conditions is None or self.condition_names is None:
            raise ValueError("Cannot parse contrast without condition_names")

        weights = np.zeros(self.n_conditions)

        # Simple parser: split by +/- and assign weights
        # e.g., "face-house" -> face=+1, house=-1

        # Add leading + if needed
        if not contrast.startswith(("+", "-")):
            contrast = "+" + contrast

        # Find all terms: (+/-)name
        pattern = r"([+-])(\w+)"
        condition_names = self.condition_names  # Local var for type checker
        for match in re.finditer(pattern, contrast):
            sign = 1 if match.group(1) == "+" else -1
            name = match.group(2)

            if name not in condition_names:
                raise ValueError(
                    f"Unknown condition: {name}. Available: {condition_names}"
                )

            idx = condition_names.index(name)
            weights[idx] = sign

        return weights

    def repool(self, param: str) -> PooledData:
        """Re-extract different parameter from saved fitted state.

        Args:
            param: Parameter to extract (e.g., 'residual', 't').

        Returns:
            New PooledData with the requested parameter.

        Raises:
            ValueError: If no fitted state was saved.
        """
        if self.fitted_state is None:
            raise ValueError(
                "No fitted state saved. Use save_fitted=True when calling pool()."
            )

        # Extract the requested parameter from fitted state
        new_data = self._extract_param(param)
        return PooledData(
            data=new_data,
            param=param,
            condition_names=self.condition_names,
            subject_ids=self.subject_ids,
            fitted_state=self.fitted_state,
            save_path=self.save_path,
        )

    def _extract_param(self, param: str) -> NDArray:
        """Extract parameter from fitted state."""
        # This depends on how fitted_state is structured
        # For now, assume it's a list of dicts per subject
        if isinstance(self.fitted_state, list):
            return np.stack([s.get(param) for s in self.fitted_state])
        raise ValueError(f"Cannot extract {param} from fitted state")

    def save(self, path: str) -> None:
        """Save pooled data to disk.

        Args:
            path: Output path (directory or .npz file).
        """
        import json

        path_obj = Path(path)

        if path_obj.suffix == ".npz":
            # np.savez requires arrays, so we save metadata separately
            # to handle None values properly
            np.savez(
                path_obj,
                data=self.data,
            )
            # Save metadata as JSON alongside
            meta_path = path_obj.with_suffix(".json")
            meta = {
                "param": self.param,
                "condition_names": self.condition_names,
                "subject_ids": self.subject_ids,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
        else:
            path_obj.mkdir(parents=True, exist_ok=True)
            np.save(path_obj / "pooled_data.npy", self.data)
            # Save metadata
            meta = {
                "param": self.param,
                "condition_names": self.condition_names,
                "subject_ids": self.subject_ids,
            }
            with open(path_obj / "metadata.json", "w") as f:
                json.dump(meta, f)

    @classmethod
    def load(cls, path: str) -> PooledData:
        """Load pooled data from disk.

        Args:
            path: Path to saved data.

        Returns:
            Loaded pooled data.
        """
        import json

        path_obj = Path(path)

        if path_obj.suffix == ".npz":
            loaded = np.load(path_obj, allow_pickle=True)
            # Load metadata from companion JSON file
            meta_path = path_obj.with_suffix(".json")
            with open(meta_path) as f:
                meta = json.load(f)
            return cls(
                data=loaded["data"],
                param=meta["param"],
                condition_names=meta.get("condition_names"),
                subject_ids=meta.get("subject_ids"),
                save_path=str(path_obj),
            )
        data = np.load(path_obj / "pooled_data.npy")
        with open(path_obj / "metadata.json") as f:
            meta = json.load(f)
        return cls(
            data=data,
            param=meta["param"],
            condition_names=meta.get("condition_names"),
            subject_ids=meta.get("subject_ids"),
            save_path=str(path_obj),
        )

    def __repr__(self) -> str:
        shape_str = f"({self.n_subjects}"
        if self.n_conditions:
            shape_str += f", {self.n_conditions}"
        shape_str += f", {self.n_voxels})"
        return f"PooledData(param='{self.param}', shape={shape_str})"


@dataclass
class StatResult:
    """Result of statistical test.

    Holds statistical maps and provides thresholding utilities.
    """

    t_map: NDArray | None = None
    f_map: NDArray | None = None
    p_map: NDArray | None = None
    contrast: str | None = None
    df: int | None = None

    def threshold(self, method: str = "fdr", alpha: float = 0.05) -> StatResult:
        """Apply multiple comparison correction.

        Args:
            method: Correction method: 'fdr', 'bonferroni', or 'uncorrected'.
            alpha: Significance threshold.

        Returns:
            New result with thresholded maps.
        """
        from scipy.stats import false_discovery_control

        if self.p_map is None:
            raise ValueError("No p-values to threshold")

        if method == "fdr":
            # Benjamini-Hochberg FDR. false_discovery_control returns adjusted
            # p-values (scipy >= 1.11, which the project already requires). No
            # bare-except fallback to uncorrected thresholding: a failure here
            # (NaNs, shape errors) must surface rather than silently return
            # uncorrected significance while the user believes FDR was applied.
            adjusted_p = false_discovery_control(self.p_map.ravel())
            mask = adjusted_p.reshape(self.p_map.shape) < alpha
        elif method == "bonferroni":
            adjusted_p = np.minimum(self.p_map * self.p_map.size, 1.0)
            mask = adjusted_p < alpha
        elif method == "uncorrected":
            mask = self.p_map < alpha
        else:
            raise ValueError(f"Unknown method: {method}")

        # Apply mask
        new_t = self.t_map * mask if self.t_map is not None else None
        new_f = self.f_map * mask if self.f_map is not None else None

        return StatResult(
            t_map=new_t,
            f_map=new_f,
            p_map=self.p_map,
            contrast=self.contrast,
            df=self.df,
        )

    def to_nifti(self, path: str, mask=None) -> None:
        """Save as NIfTI file.

        Args:
            path: Output path.
            mask: Mask to use for reconstruction.
        """
        # Placeholder - would need mask to reconstruct 3D
        raise NotImplementedError("to_nifti requires mask integration")

    def __repr__(self) -> str:
        if self.t_map is not None:
            stat_type = "t-test"
            max_val = float(np.nanmax(np.abs(self.t_map)))
        elif self.f_map is not None:
            stat_type = "F-test"
            max_val = float(np.nanmax(self.f_map))
        else:
            stat_type = "unknown"
            max_val = 0

        return f"StatResult({stat_type}, max={max_val:.2f}, contrast='{self.contrast}')"


class ResultDict(dict):
    """Dictionary of StatResults, one per contrast.

    Provides convenience methods for batch operations.
    """

    def threshold_all(self, method: str = "fdr", alpha: float = 0.05) -> ResultDict:
        """Apply thresholding to all results.

        Args:
            method: Correction method: 'fdr', 'bonferroni', or 'uncorrected'.
            alpha: Significance threshold.

        Returns:
            New dict with thresholded results.
        """
        return ResultDict(
            {k: v.threshold(method=method, alpha=alpha) for k, v in self.items()}
        )

    def __repr__(self) -> str:
        contrasts = list(self.keys())
        return f"ResultDict({contrasts})"


__all__ = ["PooledData", "ResultDict", "StatResult"]
