import numpy as np
# Pascal-Diamond Hashing (PDH)
def pascal_diamond_hash(data):
    """Hash data using Pascal-Diamond weighting."""
    prime_mod = 2**61 - 1  # Large prime for hashing
    hash_val = 0

    for i, value in enumerate(data):
        hash_val = (hash_val + (value * (i + 1))) % prime_mod

    return format(hash_val & 0xFFFFFFFFFFFFFFFF, "x")
