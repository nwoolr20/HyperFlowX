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

# 🚀 Optimized Parallel QuickSort
def quicksort(arr):
    """Optimized QuickSort with parallel execution."""
    if len(arr) <= 10:
        return insertion_sort(arr)
    
    pivot = quantum_pivot(arr)
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(quicksort, left)
        right_future = executor.submit(quicksort, right)
        return np.concatenate((left_future.result(), middle, right_future.result()))

# 🚀 Optimized Parallel MergeSort
def mergesort(arr):
    """Optimized MergeSort with parallel execution."""
    if len(arr) <= 1000:
        return np.sort(arr)

    mid = len(arr) // 2
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(mergesort, arr[:mid])
        right_future = executor.submit(mergesort, arr[mid:])
        left, right = left_future.result(), right_future.result()
    
    return np.concatenate((left, right))

# 🚀 Hybrid Sort (Automatically Selects Best Method)
def hybrid_sort(arr):
    arr = np.array(arr)
    if len(arr) < 1000:
        return insertion_sort(arr)
    elif len(arr) < 10_000:
        return quicksort(arr)
    else:
        return mergesort(arr)
