import numpy as np
import numba

# 🚀 Optimized Parallel Matrix Multiplication (SIMD-Accelerated)
@numba.njit(parallel=True, fastmath=True)
def fast_matrix_mult(A, B):
    """Performs fast parallel matrix multiplication using NumPy dot product (SIMD-accelerated)."""
    return np.dot(A, B)  # ✅ Uses NumPy’s highly optimized BLAS backend
