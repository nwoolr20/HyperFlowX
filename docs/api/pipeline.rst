Asynchronous Pipeline
=====================

Concurrent execution of multiple computational tasks.

.. automodule:: hyperflowx.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: hyperflowx.pipeline.async_pipeline

Pipeline Architecture
---------------------

The async pipeline coordinates multiple computational tasks:

* **Concurrent Execution**: Runs sorting, ML, security, and optimization tasks in parallel
* **Task Coordination**: Uses asyncio for efficient resource utilization
* **Thread Pool Integration**: CPU-bound tasks executed in thread pools
* **Error Handling**: Comprehensive exception management across tasks

Supported Operations
--------------------

The pipeline executes these tasks concurrently:

1. **Sorting Task**: Large array sorting with hybrid algorithms
2. **ML Task**: Model training with automatic algorithm selection  
3. **Security Task**: Hash computation for data integrity
4. **Optimization Task**: Matrix operations in background threads

Performance Benefits
--------------------

* **Parallelism**: Utilizes multiple CPU cores simultaneously
* **I/O Efficiency**: Non-blocking operations for better resource usage
* **Scalability**: Handles multiple concurrent workflows
* **Throughput**: Significantly higher than sequential execution

Implementation Details
----------------------

* **AsyncIO Event Loop**: Coordinates all asynchronous operations
* **ThreadPoolExecutor**: Handles CPU-intensive computations
* **Task Scheduling**: Efficient work distribution across resources
* **Memory Management**: Careful handling of large data structures

Usage Examples
--------------

.. code-block:: python

   import asyncio
   from hyperflowx.pipeline import async_pipeline

   # Run concurrent pipeline
   async def main():
       await async_pipeline()

   # Execute the pipeline
   asyncio.run(main())

Technical Specifications
------------------------

* **Concurrency Model**: Cooperative multitasking with asyncio
* **Thread Safety**: Thread-safe operations for shared resources
* **Error Recovery**: Graceful handling of task failures
* **Resource Limits**: Configurable limits for memory and CPU usage

Performance Metrics
-------------------

Compared to sequential execution:

* **Throughput**: 3-4× improvement for typical workloads
* **Latency**: Reduced time-to-completion for batch operations
* **Resource Utilization**: Better CPU and memory efficiency