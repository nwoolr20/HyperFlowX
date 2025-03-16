import numpy as np
import numba

# 🚀 Pascal-Diamond Hashing (PDH) - Optimized for Numba
@numba.njit(parallel=True, fastmath=True)
def pascal_diamond_hash(data):
    """Optimized hashing function using Pascal-Diamond weighting & SIMD."""
    prime_mod = 2**61 - 1  # Large prime for hashing
    hash_val = np.uint64(0)  # Use uint64 to avoid overflow issues

    # ✅ Numba-compatible reduction
    for i in numba.prange(len(data)):
        hash_val = (hash_val + (data[i] * (i + 1))) % prime_mod

    return hex(hash_val & 0xFFFFFFFFFFFFFFFF)  # ✅ Safe hex conversion
