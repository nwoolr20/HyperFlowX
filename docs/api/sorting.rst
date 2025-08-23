Sorting Algorithms
==================

This module provides high-performance sorting algorithms with adaptive selection.

.. automodule:: hyperflowx.sorting
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: hyperflowx.sorting.hybrid_sort

.. autofunction:: hyperflowx.sorting.adaptive_sort

.. autofunction:: hyperflowx.sorting.insertion_sort

.. autofunction:: hyperflowx.sorting.dual_pivot_quicksort

.. autofunction:: hyperflowx.sorting.radix_sort

.. autofunction:: hyperflowx.sorting.counting_sort_by_digit

Algorithm Selection
-------------------

The sorting module uses intelligent algorithm selection:

* **Small arrays** (< 50 elements): Insertion sort for optimal performance
* **Medium arrays** (50-10,000): Dual-pivot quicksort with parallel processing
* **Large arrays** (> 10,000): Radix sort for integers, quicksort for floats
* **Integer data**: Specialized radix sort for positive integers

Performance Characteristics
---------------------------

* **Time Complexity**: O(n log n) average case, O(n) best case for specialized data
* **Space Complexity**: O(log n) for in-place sorting variants
* **Parallel Processing**: Utilizes multiple cores for large datasets
* **JIT Compilation**: Numba optimization for near-native performance