Installation
============

Requirements
------------

HyperFlowX requires Python 3.8+ and the following dependencies:

* NumPy >= 2.1.3
* Numba >= 0.58.0 (for JIT compilation)
* PyTorch >= 2.0.0 (for neural networks and GPU support)
* XGBoost >= 1.7.0 (for gradient boosting)
* Scikit-learn >= 1.3.0 (for ML utilities)
* SciPy >= 1.10.0 (for scientific computing)

Installation Methods
--------------------

From Source (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/nwoolr20/HyperFlowX.git
   cd HyperFlowX
   pip install -e .

This installs HyperFlowX in development mode with all dependencies.

Using pip (Future)
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install hyperflowx

*Note: PyPI package coming soon*

Verification
------------

To verify your installation:

.. code-block:: python

   import hyperflowx
   print(f"HyperFlowX version: {hyperflowx.__version__}")

   # Test basic functionality
   import numpy as np
   data = np.random.randint(0, 100, 1000)
   sorted_data = hyperflowx.hybrid_sort(data)
   print("✅ Installation successful!")

Optional Dependencies
---------------------

For GPU acceleration:

.. code-block:: bash

   # CUDA support (if available)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

For development:

.. code-block:: bash

   pip install pytest pytest-cov hypothesis sphinx sphinx-rtd-theme
   pip install black flake8 mypy bandit

System Requirements
-------------------

**Minimum:**
* CPU: Multi-core processor (2+ cores recommended)
* RAM: 4GB (8GB+ recommended for large datasets)
* Python: 3.8+

**Recommended:**
* CPU: 8+ cores for optimal parallel performance
* RAM: 16GB+ for large-scale computations
* GPU: CUDA-compatible for neural network acceleration
* SSD: Fast storage for large dataset operations

Troubleshooting
---------------

Common installation issues:

**NumPy/Numba compatibility:**

.. code-block:: bash

   pip install --upgrade numba numpy

**CUDA/PyTorch issues:**

.. code-block:: bash

   # Install CPU-only version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

**Permission errors:**

.. code-block:: bash

   pip install --user hyperflowx

**Build failures:**

.. code-block:: bash

   # Ensure build tools are available
   pip install --upgrade setuptools wheel build