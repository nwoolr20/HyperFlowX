import numpy as np
import numba

# 🚀 Optimized Parallel Matrix Multiplication (SIMD-Accelerated)
@numba.njit(parallel=True, fastmath=True)
def fast_matrix_mult(A, B):
    """Perform fast matrix multiplication without requiring SciPy."""
    n, m = A.shape[0], B.shape[1]
    result = np.zeros((n, m))
    for i in numba.prange(n):
        for j in range(m):
            acc = 0.0
            for k in range(A.shape[1]):
                acc += A[i, k] * B[k, j]
            result[i, j] = acc
    return result
