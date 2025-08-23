"""High-performance sorting algorithms with adaptive selection.

This module provides optimized sorting implementations using Numba JIT compilation
and intelligent algorithm selection based on data characteristics.
"""

import numpy as np
import numba
from typing import Union, cast
from .monitoring import monitor_performance, log_info


# 🚀 Optimized Insertion Sort (for small arrays)
@numba.njit  # type: ignore[misc]
def insertion_sort(arr: np.ndarray) -> np.ndarray:
    """Optimized insertion sort for small arrays.

    Args:
        arr: Input array to sort

    Returns:
        Sorted array

    Note:
        Most efficient for arrays with < 50 elements.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr


# 🚀 Ultra-Fast Dual-Pivot QuickSort (Numba-Optimized)
@numba.njit  # type: ignore[misc]
def dual_pivot_quicksort(arr: np.ndarray, left: int, right: int) -> None:
    """Dual-pivot quicksort - faster than single-pivot for larger arrays."""
    if right - left < 17:  # Use insertion sort for small subarrays
        insertion_sort_range(arr, left, right)
        return

    # Choose pivots
    if arr[left] > arr[right]:
        arr[left], arr[right] = arr[right], arr[left]

    pivot1 = arr[left]
    pivot2 = arr[right]

    # Three-way partitioning
    i = left + 1
    lt = left + 1  # Elements < pivot1
    gt = right - 1  # Elements > pivot2

    while i <= gt:
        if arr[i] < pivot1:
            arr[i], arr[lt] = arr[lt], arr[i]
            lt += 1
        elif arr[i] > pivot2:
            arr[i], arr[gt] = arr[gt], arr[i]
            gt -= 1
            i -= 1  # Re-examine swapped element
        i += 1

    # Place pivots in their final positions
    arr[left], arr[lt - 1] = arr[lt - 1], arr[left]
    arr[right], arr[gt + 1] = arr[gt + 1], arr[right]

    # Recursively sort the three partitions
    dual_pivot_quicksort(arr, left, lt - 2)
    dual_pivot_quicksort(arr, lt, gt)
    dual_pivot_quicksort(arr, gt + 2, right)


@numba.njit  # type: ignore[misc]
def insertion_sort_range(arr: np.ndarray, left: int, right: int) -> None:
    """Insertion sort for a specific range."""
    for i in range(left + 1, right + 1):
        key = arr[i]
        j = i - 1
        while j >= left and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


# 🚀 Radix Sort for Integer Arrays (Ultra-Fast for specific cases)
@numba.njit  # type: ignore[misc]
def radix_sort(arr: np.ndarray) -> np.ndarray:
    """High-speed radix sort for positive integers."""
    if len(arr) <= 1:
        return arr

    # Convert to integers and handle negatives
    int_arr = np.asarray(arr, dtype=np.int64)

    # Find the maximum number to know number of digits
    max_num: int = np.max(np.abs(int_arr))
    if max_num == 0:
        return arr.astype(np.float64)

    # Do counting sort for every digit
    exp = 1
    while max_num // exp > 0:
        counting_sort_by_digit(int_arr, exp)
        exp *= 10

    return int_arr.astype(np.float64)  # type: ignore[no-any-return]


@numba.njit  # type: ignore[misc]
def counting_sort_by_digit(arr: np.ndarray, exp: int) -> None:
    """Counting sort by a specific digit."""
    n = len(arr)
    output = np.zeros(n, dtype=np.int64)
    count = np.zeros(10, dtype=np.int64)

    # Store count of occurrences
    for i in range(n):
        index: int = int((abs(arr[i]) // exp) % 10)
        count[index] += 1

    # Change count[i] so it contains actual position
    for i in range(1, 10):
        count[i] += count[i - 1]

    # Build the output array
    i = n - 1
    while i >= 0:
        index = int((abs(arr[i]) // exp) % 10)
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1

    # Copy output array to arr
    for i in range(n):
        arr[i] = output[i]


# 🚀 Intelligent Hybrid Sorting Algorithm (Optimized)
def hybrid_sort(arr: Union[np.ndarray, list]) -> np.ndarray:
    """
    Native HyperFlowX sorting algorithm with optimized path selection.

    Philosophy: This implementation prioritizes native algorithms over NumPy dependencies.
    Optimized for speed by minimizing expensive data type checks.

    Performance: Designed to be competitive with NumPy through efficient algorithm selection.
    """
    arr = cast(np.ndarray, np.array(arr, dtype=np.float64))
    n = len(arr)

    if n <= 1:
        return arr

    # For very small arrays, use insertion sort (fastest for small data)
    if n < 50:
        return insertion_sort(arr.copy())  # type: ignore[no-any-return]

    # Quick integer check - only for small arrays where it's worth it
    if n < 10000:
        # Check if array contains only small positive integers (fast path for radix sort)
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_min >= 0 and arr_max < 100000 and np.all(arr == arr.astype(np.int64)):
            # Use radix sort for positive integer data
            return radix_sort(arr.copy())  # type: ignore[no-any-return]

    # For general data or large arrays, use dual-pivot quicksort
    # Skip expensive checks for large arrays - just use the fastest general algorithm
    arr_copy = arr.copy()
    dual_pivot_quicksort(arr_copy, 0, n - 1)
    return arr_copy


# 🚀 Alternative: Hybrid with Adaptive Fallback (for maximum performance when needed)
@monitor_performance(
    operation_name="adaptive_sort", include_args=True, include_result=True
)
def adaptive_sort(
    arr: Union[np.ndarray, list], allow_numpy_fallback: bool = False
) -> np.ndarray:
    """
    Adaptive sorting with optional NumPy fallback for maximum performance.

    Args:
        arr: Input array to sort
        allow_numpy_fallback: If True, use NumPy for very large arrays where
                            raw performance is more important than algorithmic purity
    """
    arr = cast(np.ndarray, np.array(arr, dtype=np.float64))
    n = len(arr)

    if n <= 1:
        return arr

    # Always use native algorithms for reasonable sizes
    if n < 50000:
        return hybrid_sort(arr)

    # For very large arrays, user can choose performance vs purity
    elif allow_numpy_fallback:
        # Acknowledge the trade-off: performance vs algorithmic independence
        return np.sort(arr)  # type: ignore[no-any-return]
    else:
        # Stay true to native implementation
        return hybrid_sort(arr)
