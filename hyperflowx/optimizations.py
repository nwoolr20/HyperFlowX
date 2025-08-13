import numpy as np
import numba

# 🚀 Optimized Matrix Multiplication (Simplified for Numba compatibility)
@numba.njit(parallel=True, fastmath=True, cache=True)
def fast_matrix_mult(A, B):
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
    
    return result

# 🚀 Alternative fast implementation for small matrices
@numba.njit(fastmath=True, cache=True)
def fast_matrix_mult_small(A, B):
    """Fast matrix multiplication for smaller matrices."""
    n, k, m = A.shape[0], A.shape[1], B.shape[1]
    result = np.zeros((n, m), dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            acc = 0.0
            for ki in range(k):
                acc += A[i, ki] * B[ki, j]
            result[i, j] = acc
    
    return result

# 🚀 Adaptive Matrix Multiplication
def adaptive_matrix_mult(A, B):
    """Choose the best matrix multiplication method based on size."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    
    # For very small matrices, use simple implementation
    if A.shape[0] * A.shape[1] * B.shape[1] < 1000000:  # 100x100x100
        return fast_matrix_mult_small(A, B)
    
    # For medium to large matrices, use parallel implementation
    elif A.shape[0] * A.shape[1] * B.shape[1] < 100000000:  # 1000x1000x100
        return fast_matrix_mult(A, B)
    
    # For very large matrices, fall back to NumPy's optimized BLAS
    else:
        return np.dot(A, B)
