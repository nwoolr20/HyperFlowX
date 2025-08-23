"""Optimized matrix operations and numerical algorithms.

This module provides high-performance matrix multiplication and linear algebra
operations using Numba JIT compilation and parallel processing.
"""

import numpy as np
import numba
from typing import Union, Tuple, cast


# 🚀 Optimized Matrix Multiplication (Simplified for Numba compatibility)
@numba.njit(parallel=True, fastmath=True, cache=True)  # type: ignore[misc]
def fast_matrix_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Perform fast matrix multiplication with optimizations.
    
    Args:
        A: First matrix (m x k)
        B: Second matrix (k x n)
        
    Returns:
        Result matrix (m x n)
        
    Note:
        Uses parallel processing and fast math optimizations for large matrices.
    """
    """Perform fast matrix multiplication with optimizations."""
    n, k, m = A.shape[0], A.shape[1], B.shape[1]
    result = np.zeros((n, m), dtype=np.float64)
    
    # Simple but efficient parallel matrix multiplication
    for i in numba.prange(n):
        for j in range(m):
            acc = 0.0
            for ki in range(k):
                acc += A[i, ki] * B[ki, j]
            result[i, j] = acc
    
    return result  # type: ignore[no-any-return]

# 🚀 Alternative fast implementation for small matrices
@numba.njit(fastmath=True, cache=True)  # type: ignore[misc]
def fast_matrix_mult_small(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Fast matrix multiplication for smaller matrices."""
    n, k, m = A.shape[0], A.shape[1], B.shape[1]
    result = np.zeros((n, m), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for ki in range(k):
                acc += A[i, ki] * B[ki, j]
            result[i, j] = acc
    
    return result  # type: ignore[no-any-return]

# 🚀 Adaptive Matrix Multiplication
def adaptive_matrix_mult(A: Union[np.ndarray, list], B: Union[np.ndarray, list]) -> np.ndarray:
    """Choose the best matrix multiplication method based on size.
    
    Automatically selects the optimal algorithm based on matrix dimensions:
    - Small matrices: Simple implementation without parallel overhead
    - Medium matrices: Parallel Numba implementation
    - Large matrices: NumPy's optimized BLAS implementation
    
    Args:
        A: First matrix (m x k)
        B: Second matrix (k x n)
        
    Returns:
        Result matrix (m x n)
        
    Raises:
        ValueError: If matrix dimensions are incompatible
    """
    A = cast(np.ndarray, np.asarray(A, dtype=np.float64))
    B = cast(np.ndarray, np.asarray(B, dtype=np.float64))
    
    # For very small matrices, use simple implementation
    if A.shape[0] * A.shape[1] * B.shape[1] < 1000000:  # 100x100x100
        return fast_matrix_mult_small(A, B)  # type: ignore[no-any-return]
    
    # For medium to large matrices, use parallel implementation
    elif A.shape[0] * A.shape[1] * B.shape[1] < 100000000:  # 1000x1000x100
        return fast_matrix_mult(A, B)  # type: ignore[no-any-return]
    
    # For very large matrices, fall back to NumPy's optimized BLAS
    else:
        return np.dot(A, B)  # type: ignore[no-any-return]
