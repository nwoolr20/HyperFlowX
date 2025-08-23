Machine Learning
================

AI-powered machine learning with dynamic model selection.

.. automodule:: hyperflowx.ml_model
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: hyperflowx.ml_model.train_hyperflowx

Classes
-------

.. autoclass:: hyperflowx.ml_model.NeuralNet
   :members:
   :undoc-members:
   :show-inheritance:

Model Selection Logic
--------------------

The ML module automatically selects the optimal algorithm:

* **Small datasets** (≤50 features): XGBoost regression for speed and accuracy
* **Large datasets** (>50 features): Neural networks for complex pattern learning
* **GPU Acceleration**: Automatic CUDA utilization when available
* **Performance Optimization**: Adaptive hyperparameter selection

Supported Models
----------------

XGBoost Regressor
~~~~~~~~~~~~~~~~~
* Fast training on structured data
* Built-in feature importance
* Robust to overfitting
* Optimal for tabular data

Neural Networks  
~~~~~~~~~~~~~~~
* Deep learning capabilities
* GPU acceleration support
* Handles high-dimensional data
* Automatic differentiation

Usage Examples
--------------

.. code-block:: python

   import numpy as np
   from hyperflowx.ml_model import train_hyperflowx

   # Small dataset - uses XGBoost
   X_small = np.random.rand(1000, 20)  
   y_small = np.random.rand(1000)
   model_xgb = train_hyperflowx(X_small, y_small)

   # Large dataset - uses Neural Network
   X_large = np.random.rand(1000, 100)
   y_large = np.random.rand(1000)  
   model_nn = train_hyperflowx(X_large, y_large, use_cuda=True)