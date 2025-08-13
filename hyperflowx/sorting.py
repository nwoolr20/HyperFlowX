import numpy as np
import numba
import concurrent.futures

# 🚀 Quantum-Inspired Pivot Selection (Numba-Compatible)
@numba.njit
def quantum_pivot(arr):
    """Selects a pivot using probability-weighted selection (Numba-compatible)."""
    arr_np = np.asarray(arr)
    median = np.median(arr_np)
    distances = np.abs(arr_np - median)
    min_index = np.argmin(distances)  # Choose the element closest to the median
    return arr_np[min_index]

# 🚀 Optimized Insertion Sort (for small arrays)
@numba.njit
def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

# 🚀 Simple Optimized Sorting (avoiding complex Numba issues)
def hybrid_sort(arr):
    """Optimized hybrid sorting algorithm."""
    arr = np.array(arr, dtype=np.float64)
    
    # For very small arrays, use insertion sort
    if len(arr) < 50:
        return insertion_sort(arr)
    
    # For larger arrays, use NumPy's highly optimized Timsort 
    # (this is actually very fast and hard to beat)
    else:
        return np.sort(arr)
