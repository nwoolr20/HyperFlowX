"""Cryptographic hashing functions optimized for performance.

This module provides custom hashing algorithms including the Pascal-Diamond
hash function designed for high-performance applications.
"""

import numpy as np
import numba
from typing import Union


# 🚀 Pascal-Diamond Hashing (PDH) - Optimized (removed numba for string formatting)
def pascal_diamond_hash(data: Union[bytes, np.ndarray]) -> str:
    """Hash data using optimized Pascal-Diamond weighting.
    
    The Pascal-Diamond hash uses weighted accumulation based on Pascal's triangle
    coefficients to create a unique fingerprint of the input data.
    
    Args:
        data: Input data to hash (bytes or numpy array)
        
    Returns:
        Hexadecimal string representation of the hash
        
    Note:
        This is a custom hash function designed for performance testing.
        For cryptographic security, use established algorithms like SHA-256.
    """
    prime_mod = 2**61 - 1  # Large prime for hashing
    
    # Convert to numpy array for vectorized operations
    if isinstance(data, bytes):
        data_array = np.frombuffer(data, dtype=np.uint8)
    else:
        data_array = np.asarray(data, dtype=np.uint8)
    
    if len(data_array) == 0:
        return "0"
    
    # For small arrays, use vectorized operation directly
    if len(data_array) <= 100000:
        # Create position weights (i + 1)
        weights = np.arange(1, len(data_array) + 1, dtype=np.int64)
        # Vectorized computation
        hash_val = np.sum(data_array.astype(np.int64) * weights) % prime_mod
    else:
        # For large arrays, use chunking to prevent memory issues
        chunk_size = 100000
        hash_val = 0
        
        for start in range(0, len(data_array), chunk_size):
            end = min(start + chunk_size, len(data_array))
            chunk = data_array[start:end]
            weights = np.arange(start + 1, end + 1, dtype=np.int64)
            weighted_sum = np.sum(chunk.astype(np.int64) * weights)
            hash_val = (hash_val + weighted_sum) % prime_mod
    
    return format(int(hash_val) & 0xFFFFFFFFFFFFFFFF, "x")
