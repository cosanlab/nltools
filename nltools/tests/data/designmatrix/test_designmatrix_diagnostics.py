import matplotlib

matplotlib.use("Agg")  # headless backend for plot tests

import numpy as np
import pytest
from nltools.data import Adjacency
from nltools.data.designmatrix import DesignMatrix


class TestDesignMatrixDiagnostics:
    """
    Test collinearity diagnostics and cleaning.

    Behavioral contract:
    - vif() computes variance inflation factor
    - clean() removes highly correlated columns
    - Correlation threshold configurable
    """

    # NOTE: make sure this implementation is also as efficient as possible using built-in polars functionality
    def test_vif_computes_variance_inflation_factor(self):
        """
        VIF measures collinearity: VIF > 10 indicates problematic correlation.

        Expected behavior:
        - Returns array of VIF values (one per column)
        - With perfect collinearity, returns None and prints error

        Use case: Detect multicollinearity before regression
        """
        # Create intentionally collinear columns (perfect collinearity)
        dm_perfect = DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [2, 4, 6, 8, 10],  # b = 2*a, perfect collinearity
                "c": [1, 1, 2, 2, 3],  # Moderately correlated
            },
            sampling_freq=1,
        )

        # Perfect collinearity should return None with error message
        vifs_perfect = dm_perfect.vif(exclude_confounds=True)
        assert vifs_perfect is None, "Perfect collinearity should return None"

        # Create data with high but not perfect correlation
        dm_high = DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.9, 4.1, 5.8, 8.2, 9.9],  # High but not perfect correlation
                "c": [1, 1, 2, 2, 3],
            },
            sampling_freq=1,
        )

        vifs_high = dm_high.vif(exclude_confounds=True)
        assert vifs_high is not None, (
            "Should return VIF values for non-perfect correlation"
        )
        assert len(vifs_high) == 3, "Should have VIF for each column"
        # VIF should be elevated for highly correlated columns
        assert any(v > 5 for v in vifs_high), "Should detect high collinearity"

    def test_vif_excludes_polynomial_columns_by_default(self):
        """
        exclude_confounds=True should not compute VIF for polynomial columns.

        Expected behavior:
        - Polynomials skipped in VIF calculation
        - Only data columns included

        Rationale: Polynomials are expected to be correlated (e.g., linear and quadratic)
        """
        dm = DesignMatrix({"a": [1, 2, 3, 4, 5]}, sampling_freq=1)
        dm = dm.add_poly(order=2)  # Adds poly_0, poly_1, poly_2

        vifs = dm.vif(exclude_confounds=True)

        # Should only compute VIF for 'a', not polynomials
        assert len(vifs) == 1, "Should only compute VIF for non-polynomial columns"

    def test_clean_removes_highly_correlated_columns(self):
        """
        .clean() drops columns correlated above threshold.

        Expected behavior:
        - Columns with r >= thresh are dropped
        - First instance of correlated pair is kept
        - Returns new DesignMatrix

        Use case: Remove redundant regressors before fitting
        """
        dm = DesignMatrix(
            {
                "a": [1, 2, 3, 4],
                "b": [1.01, 2.01, 3.01, 4.01],  # r ≈ 1.0 with 'a'
                "c": [4, 3, 2, 1],  # Uncorrelated with a, b
            },
            sampling_freq=1,
        )

        dm_clean = dm.clean(thresh=0.95)

        # Should keep 'a' (first instance), drop 'b' (correlated duplicate)
        assert "a" in dm_clean.columns, "First instance should be kept"
        assert "b" not in dm_clean.columns, "Highly correlated column should be dropped"
        assert "c" in dm_clean.columns, "Uncorrelated column should be kept"

    def test_clean_with_lower_threshold(self):
        """
        Lower threshold should drop more columns.

        Expected behavior:
        - thresh=0.8 more aggressive than thresh=0.95
        - Moderately correlated columns also dropped
        """
        dm = DesignMatrix(
            {
                "a": [1, 2, 3, 4, 5],
                "b": [1.1, 2.2, 3.1, 3.9, 5.1],  # r ≈ 0.99 with 'a'
                "c": [2, 3, 4, 5, 6],  # r ≈ 1.0 with 'a'
            },
            sampling_freq=1,
        )

        dm_clean = dm.clean(thresh=0.8)

        # Both b and c should be dropped (highly correlated with a)
        assert "a" in dm_clean.columns
        assert dm_clean.shape[1] < dm.shape[1], "Should drop correlated columns"

    def test_clean_excludes_confounds_from_collinearity_check(self):
        """
        exclude_confounds=True should keep polynomial columns even if correlated.

        Expected behavior:
        - Polynomials not checked for collinearity
        - Preserved in output even if correlated with each other

        Rationale: Polynomial trends naturally correlated, but needed for modeling
        """
        dm = DesignMatrix({"a": [1, 2, 3, 4]}, sampling_freq=1)
        dm = dm.add_poly(order=2)  # poly_1 and poly_2 might be correlated

        dm_clean = dm.clean(exclude_confounds=True)

        # All polynomials should be retained
        assert "poly_0" in dm_clean.columns
        assert "poly_1" in dm_clean.columns
        assert "poly_2" in dm_clean.columns

    def test_clean_fillna_before_checking(self):
        """
        fill_na parameter should fill NaNs before correlation check.

        Expected behavior:
        - NaNs filled with specified value
        - Correlation computed on filled data

        Use case: Handle missing motion data before diagnostics
        """
        dm = DesignMatrix(
            {"a": [1.0, None, 3.0, 4.0], "b": [2.0, None, 6.0, 8.0]}, sampling_freq=1
        )

        dm_clean = dm.clean(fill_na=0, thresh=0.95)

        # Should successfully compute correlations after filling
        assert dm_clean.shape[0] == 4, "All rows should be preserved"


# ============================================================================
# 9. Utility Tests
# ============================================================================


class TestDesignMatrixUtilities:
    """
    Test miscellaneous utility methods.

    Behavioral contract:
    - details() returns human-readable metadata string
    - replace_data() swaps data while preserving polynomials
    - heatmap() creates visualization (tested separately)
    """

    def test_repr_shows_metadata(self):
        """
        repr(dm) should summarize metadata.

        Expected behavior:
        - Contains sampling_freq
        - Contains shape
        - Lists convolved columns
        - Lists polynomial columns

        Use case: Quick inspection of DesignMatrix state at the REPL
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=2)
        dm = dm.add_poly(0)
        dm = dm.convolve(columns=["a"])

        text = repr(dm)

        assert "sampling_freq=2" in text
        assert "shape=(3, 2)" in text or "(3, 2)" in text
        assert "poly_0" in text or "confounds" in text
        assert "convolved" in text

    def test_replace_data_keeps_metadata_and_confounds(self):
        """
        .replace_data() swaps data columns but preserves polynomial columns.

        Expected behavior:
        - Old data columns removed
        - New data columns added (with provided names)
        - Polynomial columns unchanged
        - Metadata preserved

        Use case: Substitute stimulus regressors while keeping drift terms
        """
        dm = DesignMatrix({"a": [1, 2, 3], "b": [4, 5, 6]}, sampling_freq=2)
        dm = dm.add_poly(order=0)  # Adds poly_0

        # Replace data with new columns
        new_data = np.array([[10, 20], [30, 40], [50, 60]])
        dm_replaced = dm.replace_data(new_data, column_names=["x", "y"])

        # New data columns should be present
        assert "x" in dm_replaced.columns
        assert "y" in dm_replaced.columns
        assert dm_replaced["x"].to_list() == [10, 30, 50]

        # Old data columns should be gone
        assert "a" not in dm_replaced.columns
        assert "b" not in dm_replaced.columns

        # Polynomials should be preserved
        assert "poly_0" in dm_replaced.columns
        assert dm_replaced["poly_0"].to_list() == dm["poly_0"].to_list()

        # Metadata should be preserved
        assert dm_replaced.sampling_freq == 2

    def test_replace_data_validates_row_count(self):
        """
        .replace_data() should error if new data has different number of rows.

        Expected behavior:
        - Raise ValueError if row count mismatch
        - Prevents invalid state

        Rationale: Polynomial columns have specific length, can't change
        """
        dm = DesignMatrix({"a": [1, 2, 3]}, sampling_freq=1)
        dm = dm.add_poly(0)

        # Try to replace with different row count
        new_data = np.array([[10], [20]])  # Only 2 rows, original has 3

        with pytest.raises(ValueError):
            _ = dm.replace_data(new_data, column_names=["x"])

    # NOTE: since polars dataframes don't have comprehensive plotting abilities like pandas and we don't want additional dependencies, the method should use seaborn heatmap + matplotlib under-the-hood
    def test_plot_visualization(self):
        """
        .plot() should create matplotlib visualization.

        Expected behavior:
        - Creates plot without error
        - Returns matplotlib axes object (optional)

        Note: We don't test visual output, just that it doesn't crash
        """
        dm = DesignMatrix(
            np.random.randn(20, 3), sampling_freq=1, columns=["a", "b", "c"]
        )
        dm = dm.add_poly(order=1)

        # Should not raise error
        try:
            dm.plot()
            import matplotlib.pyplot as plt

            plt.close("all")  # Clean up
        except Exception as e:
            pytest.fail(f"plot() raised unexpected error: {e}")


class TestDesignMatrixCorr:
    """`.corr()` returns a labeled nltools Adjacency (similarity matrix)."""

    @staticmethod
    def _toy() -> DesignMatrix:
        rng = np.random.default_rng(0)
        return DesignMatrix(
            rng.standard_normal((20, 3)), sampling_freq=1, columns=["a", "b", "c"]
        )

    def test_corr_returns_adjacency(self):
        """corr() yields a similarity Adjacency carrying the column labels."""
        c = self._toy().corr()
        assert isinstance(c, Adjacency)
        assert c.matrix_type == "similarity"
        assert list(c.labels) == ["a", "b", "c"]
        assert c.n_nodes == 3
        assert c.squareform().shape == (3, 3)

    def test_corr_columns_subset(self):
        """columns= restricts the correlation to a subset of regressors."""
        c = self._toy().corr(columns=["a", "b"])
        assert c.n_nodes == 2
        assert list(c.labels) == ["a", "b"]

    def test_corr_spearman(self):
        """metric='spearman' is supported alongside the default pearson."""
        c = self._toy().corr(metric="spearman")
        assert c.squareform().shape == (3, 3)

    def test_corr_invalid_metric(self):
        """An unknown metric raises a helpful ValueError."""
        with pytest.raises(ValueError):
            self._toy().corr(metric="bogus")


class TestDesignMatrixPlotMethods:
    """`.plot(method=...)` dispatches to matrix / timeseries / corr views."""

    @staticmethod
    def _toy() -> DesignMatrix:
        rng = np.random.default_rng(1)
        return DesignMatrix(
            rng.standard_normal((20, 3)), sampling_freq=1, columns=["a", "b", "c"]
        )

    def test_plot_matrix_default(self):
        """Default method is the SPM-style heatmap, returning a Figure."""
        from matplotlib.figure import Figure

        fig = self._toy().plot()
        assert isinstance(fig, Figure)

    def test_plot_timeseries(self):
        """method='timeseries' draws one line per regressor column."""
        from matplotlib.figure import Figure

        fig = self._toy().plot(method="timeseries")
        assert isinstance(fig, Figure)
        assert len(fig.axes[0].lines) == 3

    def test_plot_timeseries_columns_subset(self):
        """columns= restricts which regressors are drawn."""
        fig = self._toy().plot(method="timeseries", columns=["a"])
        assert len(fig.axes[0].lines) == 1

    def test_plot_timeseries_ax_overlay(self):
        """Passing ax= overlays onto the caller's axis and returns its figure."""
        import matplotlib.pyplot as plt

        dm = self._toy()
        _, ax = plt.subplots()
        out = dm.plot(method="timeseries", columns=["a"], ax=ax)
        dm.plot(method="timeseries", columns=["b"], ax=ax)
        assert out is ax.figure
        assert len(ax.lines) == 2

    def test_plot_corr(self):
        """method='corr' renders a correlation heatmap with a 1.0 diagonal."""
        from matplotlib.figure import Figure

        fig = self._toy().plot(method="corr")
        assert isinstance(fig, Figure)
        arr = np.asarray(fig.axes[0].collections[0].get_array()).reshape(3, 3)
        assert np.allclose(np.diag(arr), 1.0)

    def test_plot_invalid_method(self):
        """An unknown method raises ValueError listing the valid options."""
        with pytest.raises(ValueError):
            self._toy().plot(method="bogus")

    def test_plot_save(self, tmp_path):
        """save= writes the figure to disk."""
        out = tmp_path / "dm.png"
        self._toy().plot(method="corr", save=str(out))
        assert out.exists()
