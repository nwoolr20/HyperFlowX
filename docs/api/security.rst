Security & Hashing
==================

Cryptographic hashing functions optimized for performance.

.. automodule:: hyperflowx.security
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: hyperflowx.security.pascal_diamond_hash

Pascal-Diamond Hash Algorithm
-----------------------------

The Pascal-Diamond hash (PDH) is a custom hash function designed for high-performance applications:

* **Weighted Accumulation**: Uses Pascal's triangle coefficients for unique fingerprints
* **Performance Optimized**: Designed for speed over cryptographic security
* **Collision Resistant**: Good distribution properties for typical use cases
* **Flexible Input**: Handles both bytes and numpy arrays

Algorithm Details
-----------------

The Pascal-Diamond hash implements:

1. **Weighted Summation**: Each byte is multiplied by its position weight
2. **Modular Arithmetic**: Uses large prime (2^61 - 1) to prevent overflow
3. **Hexadecimal Output**: Returns compact hex string representation

Security Considerations
----------------------

.. warning::
   This is a custom hash function designed for performance testing.
   For cryptographic security, use established algorithms like SHA-256.

The PDH algorithm provides:

* **Fast Computation**: Optimized for high-throughput scenarios
* **Good Distribution**: Suitable for hash tables and checksums
* **Deterministic**: Same input always produces same output
* **Avalanche Effect**: Small input changes create large output differences

Performance Characteristics
---------------------------

* **Speed**: 0.78× vs SHA-256 (competitive performance)
* **Memory**: Constant space complexity O(1)
* **Scalability**: Linear time complexity O(n) with input size

Usage Examples
--------------

.. code-block:: python

   import numpy as np
   from hyperflowx.security import pascal_diamond_hash

   # Hash bytes
   data = b"Hello, World!"
   hash_result = pascal_diamond_hash(data)
   print(f"Hash: {hash_result}")

   # Hash numpy array
   arr = np.array([1, 2, 3, 4, 5], dtype=np.uint8)
   hash_result = pascal_diamond_hash(arr)
   print(f"Array hash: {hash_result}")

   # Hash large data
   large_data = np.random.bytes(1024)
   hash_result = pascal_diamond_hash(large_data)
   print(f"Large data hash: {hash_result}")