import asyncio
import concurrent.futures
import numpy as np
import torch
from typing import Any

from hyperflowx.sorting import hybrid_sort
from hyperflowx.ml_model import train_hyperflowx
from hyperflowx.optimizations import fast_matrix_mult
from hyperflowx.security import pascal_diamond_hash


# 🚀 Asynchronous Pipeline Execution
async def async_pipeline() -> None:
    """Runs sorting, ML, security, and optimization tasks in parallel."""

    # 🔹 Define async tasks
    async def sorting_task(arr: np.ndarray) -> np.ndarray:
        return hybrid_sort(arr)

    async def ml_task(X: np.ndarray, y: np.ndarray) -> Any:
        return train_hyperflowx(X, y)

    async def security_check_task(data: bytes) -> str:
        return pascal_diamond_hash(data)

    async def background_optimization_task(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, fast_matrix_mult, A, B)

    # 🔹 Generate test data
    arr = np.random.randint(0, 10000, 1000000)  # Large dataset for sorting
    X = np.random.rand(1000, 10)  # ML input data
    y = np.random.rand(1000)  # ML target labels
    security_data = np.random.bytes(256)  # Hashing data
    A, B = np.random.rand(512, 512), np.random.rand(512, 512)  # Matrix multiplication

    # 🚀 Run **all** tasks concurrently
    sorted_arr, model, hash_result, opt_result = await asyncio.gather(
        sorting_task(arr),
        ml_task(X, y),
        security_check_task(security_data),
        background_optimization_task(A, B),
    )

    print("✅ Sorting + ML + Security + Optimization completed in parallel!")


# 🚀 Run the async pipeline
if __name__ == "__main__":
    asyncio.run(async_pipeline())
