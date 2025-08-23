HyperFlowX Documentation
========================

**A High-Performance Computing Library with AI-Powered Optimizations**

HyperFlowX is an advanced computational library that combines optimized algorithms for sorting, machine learning, matrix operations, and cryptographic hashing. Built with performance in mind, it leverages cutting-edge techniques including parallel processing, GPU acceleration, and AI-driven model selection.

🎯 Key Features
---------------

* **Hybrid Sorting**: Quantum-inspired pivot selection with adaptive algorithm switching
* **AI-Powered ML**: Dynamic model selection between XGBoost and Neural Networks  
* **Optimized Matrix Operations**: SIMD-accelerated parallel matrix multiplication
* **Pascal-Diamond Hashing**: Custom cryptographic hashing algorithm
* **Asynchronous Pipeline**: Concurrent execution of multiple computational tasks

⚡ Performance Highlights
-------------------------

* Adaptive sorting algorithms that automatically select the best method based on data size
* GPU-accelerated neural networks with CUDA support
* Parallel matrix multiplication using Numba JIT compilation  
* Custom hashing algorithms optimized for specific use cases

📚 Quick Start
--------------

.. code-block:: python

   import numpy as np
   from hyperflowx import hybrid_sort, train_hyperflowx, fast_matrix_mult, pascal_diamond_hash

   # Sorting
   data = np.random.randint(0, 1000, 10000)
   sorted_data = hybrid_sort(data)

   # Machine Learning
   X = np.random.rand(1000, 20)
   y = np.random.rand(1000)
   model = train_hyperflowx(X, y)

   # Matrix Multiplication
   A = np.random.rand(512, 512)
   B = np.random.rand(512, 512)
   result = fast_matrix_mult(A, B)

   # Hashing
   data = np.random.bytes(256)
   hash_result = pascal_diamond_hash(data)

📋 Table of Contents
--------------------

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   installation
   quickstart
   examples
   performance

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/hyperflowx
   api/sorting
   api/ml_model
   api/optimizations
   api/security
   api/pipeline

.. toctree::
   :maxdepth: 2
   :caption: Development:

   contributing
   testing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

