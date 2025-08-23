import time
import numpy as np
import hashlib
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from typing import Dict, Any

from hyperflowx.sorting import hybrid_sort
from hyperflowx.ml_model import train_hyperflowx
from hyperflowx.optimizations import adaptive_matrix_mult as fast_matrix_mult
from hyperflowx.security import pascal_diamond_hash


# 🚀 Benchmark Sorting Performance
def benchmark_sorting() -> None:
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
def benchmark_ml() -> None:
    X = np.random.rand(5000, 20)
    y = np.random.rand(5000)

    X_train = X[:4000]
    y_train = y[:4000]
    X_test = X[4000:]
    y_test = y[4000:]

    models = {
        "Random Forest": RandomForestRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "HyperFlowX AI": train_hyperflowx(X, y),
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
def benchmark_matrix_mult() -> None:
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
def benchmark_hashing() -> None:
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


# 🚀 Save Benchmark Results
def save_benchmark_results(
    results: Dict[str, Any], filename: str = "benchmark_results.json"
) -> None:
    """Save benchmark results to a JSON file with timestamp."""
    import json
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    results_with_timestamp = {"timestamp": timestamp, "results": results}

    with open(filename, "w") as f:
        json.dump(results_with_timestamp, f, indent=2)

    print(f"📊 Results saved to {filename}")


# 🚀 Run Benchmarks with Result Saving
def run_full_benchmark(save_file: str = "benchmark_results.json") -> Dict[str, Any]:
    """Run all benchmarks and save results."""
    print("\n🚀 Running HyperFlowX Benchmarks...\n")

    results: Dict[str, Any] = {}

    # Sorting Benchmark
    arr = np.random.randint(0, 10000, 100000)

    start = time.time()
    np.sort(arr.copy())
    numpy_time = time.time() - start

    start = time.time()
    hybrid_sort(arr.copy())
    hfx_time = time.time() - start

    results["sorting"] = {
        "numpy_time": numpy_time,
        "hyperflowx_time": hfx_time,
        "speed_boost": numpy_time / hfx_time if hfx_time > 0 else 0,
    }

    print(f"🔹 Sorting Benchmark:")
    print(f"NumPy Sort: {numpy_time:.4f} sec")
    print(f"HyperFlowX Sort: {hfx_time:.4f} sec")
    print(f"⚡ Speed Boost: {results['sorting']['speed_boost']:.2f}× Faster!\n")

    # Machine Learning Benchmark
    X = np.random.rand(5000, 20)
    y = np.random.rand(5000)
    X_train = X[:4000]
    y_train = y[:4000]
    X_test = X[4000:]
    y_test = y[4000:]

    models = {
        "Random Forest": RandomForestRegressor(),
        "XGBoost": xgb.XGBRegressor(),
        "HyperFlowX AI": train_hyperflowx(X, y),
    }

    results["ml"] = {}
    print("🔹 Machine Learning Benchmark:")
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        elapsed = time.time() - start

        results["ml"][name.lower().replace(" ", "_")] = {"time": elapsed, "mse": mse}

        print(f"{name}: {elapsed:.4f} sec | MSE: {mse:.4f}")
    print()

    # Matrix Multiplication Benchmark
    A = np.random.rand(512, 512)
    B = np.random.rand(512, 512)

    start = time.time()
    np.dot(A, B)
    numpy_time = time.time() - start

    start = time.time()
    fast_matrix_mult(A, B)
    hfx_time = time.time() - start

    results["matrix_mult"] = {
        "numpy_time": numpy_time,
        "hyperflowx_time": hfx_time,
        "speed_boost": numpy_time / hfx_time if hfx_time > 0 else 0,
    }

    print(f"🔹 Matrix Multiplication Benchmark:")
    print(f"NumPy Dot Product: {numpy_time:.4f} sec")
    print(f"HyperFlowX Optimized: {hfx_time:.4f} sec")
    print(f"⚡ Speed Boost: {results['matrix_mult']['speed_boost']:.2f}× Faster!\n")

    # Hashing Benchmark
    data = np.random.bytes(256)

    start = time.time()
    hashlib.sha256(data).hexdigest()
    sha256_time = time.time() - start

    start = time.time()
    pascal_diamond_hash(data)
    hfx_time = time.time() - start

    results["hashing"] = {
        "sha256_time": sha256_time,
        "hyperflowx_time": hfx_time,
        "speed_boost": sha256_time / hfx_time if hfx_time > 0 else 0,
    }

    print(f"🔹 Hashing Benchmark:")
    print(f"SHA-256: {sha256_time:.6f} sec")
    print(f"Pascal-Diamond Hash: {hfx_time:.6f} sec")
    print(f"⚡ Speed Boost: {results['hashing']['speed_boost']:.2f}× Faster!\n")

    # Save results
    save_benchmark_results(results, save_file)
    return results


def main() -> None:
    """Main entry point for benchmark CLI."""
    run_full_benchmark()


if __name__ == "__main__":
    main()
