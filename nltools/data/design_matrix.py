"""
Design Matrix module - Polars-based implementation.

DesignMatrix is a class for representing design matrices with methods for data
processing (convolution, upsampling, downsampling) and intelligent concatenation
(automatically handling polynomial terms across runs).

Uses Polars for efficient DataFrame operations while maintaining compatibility
with pandas-based workflows.

For backward compatibility:
- Design_Matrix (with underscore) is an alias for DesignMatrix
- Design_Matrix_Series is deprecated (was pandas-specific, not needed with Polars)

Note: design_matrix_old.py contains the original pandas implementation for reference.
"""

from .design_matrix_new import DesignMatrix, Design_Matrix, Design_Matrix_Series

__all__ = ["DesignMatrix", "Design_Matrix", "Design_Matrix_Series"]
