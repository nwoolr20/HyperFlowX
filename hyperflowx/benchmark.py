import time
import numpy as np
import hashlib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from hyperflowx.sorting import hybrid_sort
from hyperflowx.ml_model import train_hyperflowx
from hyperflowx.optimizations import fast_matrix_mult
from hyperflowx.security import pascal_diamond_hash

# 🚀 Benchmark Sorting Performance
def benchmark_sorting():
    arr = np.random.randint(0, 1000000, 10_000_000)  # Large dataset

    # NumPy Timsort (Reference)
    start = time.time()
    np.sort(arr)
    numpy_time = time.time() - start

    # HyperFlowX Sorting
    start = time.time()
    hybrid_sort(arr)
    hyperflowx_time = time.time() - start

    print(f"🔹 Sorting Benchmark:")
    print(f"NumPy Sort: {numpy_time:.4f} sec")
    print(f"HyperFlowX Sort: {hyperflowx_time:.4f} sec")
    print(f"⚡ Speed Boost: {numpy_time / hyperflowx_time:.2f}× Faster!\n")

# 🚀 Benchmark Machine Learning Performance
def benchmark_ml():
    X = np.random.rand(5000, 20)
    y = np.random.rand(5000)

    X_train = X[:4000]
    y_train = y[:4000]
    X_test = X[4000:]
    y_test = y[4000:]

    models = {
        "Random Forest": RandomForestRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "HyperFlowX AI": train_hyperflowx(X, y)
    }

    print("🔹 Machine Learning Benchmark:")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        elapsed = time.time() - start
        print(f"{name}: {elapsed:.4f} sec | MSE: {mse:.4f}")

# 🚀 Benchmark Matrix Multiplication
def benchmark_matrix_mult():
    A = np.random.rand(1024, 1024)
    B = np.random.rand(1024, 1024)

    # NumPy Dot Product
    start = time.time()
    np.dot(A, B)
    numpy_time = time.time() - start

    # HyperFlowX Optimized Multiplication
    start = time.time()
    fast_matrix_mult(A, B)
    hyperflowx_time = time.time() - start

    print(f"🔹 Matrix Multiplication Benchmark:")
    print(f"NumPy Dot Product: {numpy_time:.4f} sec")
    print(f"HyperFlowX Optimized: {hyperflowx_time:.4f} sec")
    print(f"⚡ Speed Boost: {numpy_time / hyperflowx_time:.2f}× Faster!\n")

# 🚀 Benchmark Hashing Algorithms
def benchmark_hashing():
    data = np.random.bytes(256)

    # SHA-256 Benchmark
    start = time.time()
    hashlib.sha256(data).hexdigest()
    sha256_time = time.time() - start

    # Pascal-Diamond Hashing Benchmark
    start = time.time()
    pascal_diamond_hash(data)
    hfx_time = time.time() - start

    print(f"🔹 Hashing Benchmark:")
    print(f"SHA-256: {sha256_time:.6f} sec")
    print(f"Pascal-Diamond Hash: {hfx_time:.6f} sec")
    print(f"⚡ Speed Boost: {sha256_time / hfx_time:.2f}× Faster!\n")

# 🚀 Run Benchmarks
if __name__ == "__main__":
    print("\n🚀 Running HyperFlowX Benchmarks...\n")
    benchmark_sorting()
    benchmark_ml()
    benchmark_matrix_mult()
    benchmark_hashing()
