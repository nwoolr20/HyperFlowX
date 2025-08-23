import time
import numpy as np
import hashlib
from hyperflowx.sorting import hybrid_sort
from hyperflowx.ml_model import train_hyperflowx
from hyperflowx.optimizations import fast_matrix_mult
from hyperflowx.security import pascal_diamond_hash


# 🚀 Load Real-World Data (Simulated)
def load_data():
    X = np.random.rand(500, 10)  # ML input
    y = np.random.rand(500)  # ML labels
    arr = np.random.randint(0, 10000, 100000)  # Sorting input
    return X, y, arr


# 🚀 Run Sorting Test
def run_sorting(arr):
    start = time.time()
    sorted_arr = hybrid_sort(arr)
    elapsed = time.time() - start
    print(f"✅ Sorting Completed: {elapsed:.4f} sec")
    return sorted_arr


# 🚀 Run Machine Learning Test
def run_ml(X, y):
    start = time.time()
    model = train_hyperflowx(X, y)
    elapsed = time.time() - start
    print(f"✅ ML Model Trained: {elapsed:.4f} sec")
    return model


# 🚀 Run Security Test
def run_security():
    data = np.random.bytes(256)

    start = time.time()
    sha256_hash = hashlib.sha256(data).hexdigest()
    sha256_time = time.time() - start

    start = time.time()
    pdh_hash = pascal_diamond_hash(data)
    pdh_time = time.time() - start

    print(
        f"✅ Hashing Completed - SHA-256: {sha256_time:.6f} sec | Pascal-Diamond Hash: {pdh_time:.6f} sec"
    )


# 🚀 Main Execution
def main():
    """Main entry point for HyperFlowX examples."""
    print("\n🚀 Running HyperFlowX Example...\n")

    # Load data
    X, y, arr = load_data()

    # Run tests
    run_sorting(arr)
    run_ml(X, y)
    run_security()


if __name__ == "__main__":
    main()
