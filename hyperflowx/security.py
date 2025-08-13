import numpy as np
import numba

# 🚀 Pascal-Diamond Hashing (PDH) - Optimized (removed numba for string formatting)
def pascal_diamond_hash(data):
    """Hash data using optimized Pascal-Diamond weighting."""
    prime_mod = 2**61 - 1  # Large prime for hashing
    hash_val = 0
    
    for i in range(len(data)):
        hash_val = (hash_val + (data[i] * (i + 1))) % prime_mod
    
    return format(hash_val & 0xFFFFFFFFFFFFFFFF, "x")
