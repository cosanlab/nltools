"""
Temporary shim to maintain backward compatibility during Polars migration.

This imports the OLD Design_Matrix class so existing code doesn't break.
Once migration is complete, this will be replaced with design_matrix_new.py
"""

from .design_matrix_old import Design_Matrix, Design_Matrix_Series

__all__ = ["Design_Matrix", "Design_Matrix_Series"]
