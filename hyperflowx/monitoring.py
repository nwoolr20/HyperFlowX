"""Logging and monitoring capabilities for HyperFlowX.

This module provides comprehensive logging, performance monitoring, and metrics
collection for all HyperFlowX operations.
"""

import logging
import time
import functools
import threading
from typing import Dict, Any, Optional, Callable, Union
from datetime import datetime
import json
import os
from contextlib import contextmanager


class HyperFlowXLogger:
    """Enhanced logger with performance monitoring capabilities."""

    def __init__(self, name: str = "hyperflowx", level: int = logging.INFO) -> None:
        """Initialize the HyperFlowX logger.

        Args:
            name: Logger name
            level: Logging level
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create console handler if none exists
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Performance metrics storage
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message with optional metadata."""
        self.logger.debug(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message with optional metadata."""
        self.logger.info(message, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message with optional metadata."""
        self.logger.warning(message, extra=kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message with optional metadata."""
        self.logger.error(message, extra=kwargs)

    def critical(self, message: str, **kwargs: Any) -> None:
        """Log critical message with optional metadata."""
        self.logger.critical(message, extra=kwargs)

    def log_performance(self, operation: str, duration: float, **metadata: Any) -> None:
        """Log performance metrics for an operation.

        Args:
            operation: Name of the operation
            duration: Time taken in seconds
            **metadata: Additional performance data
        """
        with self._lock:
            if operation not in self._metrics:
                self._metrics[operation] = {
                    "total_calls": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "min_time": float("inf"),
                    "max_time": 0.0,
                    "last_call": None,
                }

            metrics = self._metrics[operation]
            metrics["total_calls"] += 1
            metrics["total_time"] += duration
            metrics["avg_time"] = metrics["total_time"] / metrics["total_calls"]
            metrics["min_time"] = min(metrics["min_time"], duration)
            metrics["max_time"] = max(metrics["max_time"], duration)
            metrics["last_call"] = datetime.now().isoformat()

            # Add metadata
            for key, value in metadata.items():
                metrics[f"last_{key}"] = value

        self.info(
            f"Performance: {operation} completed in {duration:.4f}s",
            operation=operation,
            duration=duration,
            **metadata,
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected performance metrics."""
        with self._lock:
            return self._metrics.copy()

    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        with self._lock:
            self._metrics.clear()

    def save_metrics(self, filepath: str) -> None:
        """Save metrics to a JSON file.

        Args:
            filepath: Path to save metrics file
        """
        with self._lock:
            with open(filepath, "w") as f:
                json.dump(self._metrics, f, indent=2)

        self.info(f"Metrics saved to {filepath}")


# Global logger instance
_global_logger: Optional[HyperFlowXLogger] = None


def get_logger(name: str = "hyperflowx") -> HyperFlowXLogger:
    """Get or create a HyperFlowX logger instance.

    Args:
        name: Logger name

    Returns:
        HyperFlowXLogger instance
    """
    global _global_logger
    if _global_logger is None:
        _global_logger = HyperFlowXLogger(name)
    return _global_logger


def configure_logging(
    level: Union[int, str] = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure global logging settings.

    Args:
        level: Logging level
        format_string: Custom format string
        log_file: Optional file to log to
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper())

    logger = get_logger()
    logger.logger.setLevel(level)

    # Clear existing handlers
    logger.logger.handlers.clear()

    # Create formatter
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.logger.addHandler(file_handler)


def monitor_performance(
    operation_name: Optional[str] = None,
    include_args: bool = False,
    include_result: bool = False,
    lightweight: bool = False,
) -> Callable:
    """Decorator to monitor function performance.

    Args:
        operation_name: Custom operation name (defaults to function name)
        include_args: Whether to log function arguments
        include_result: Whether to log function result type/size
        lightweight: If True, only measure time without logging (for performance-critical functions)

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if lightweight:
                # Ultra-lightweight monitoring - just timing, no logging overhead
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                # Store timing in a global dict for later retrieval if needed
                _performance_cache[func.__name__] = duration
                return result
            
            # Full monitoring with logging
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            logger = get_logger()

            # Log function start
            metadata = {}
            if include_args and args:
                metadata["args_count"] = len(args)
                if hasattr(args[0], "shape"):  # numpy array
                    metadata["input_shape"] = str(args[0].shape)
                elif hasattr(args[0], "__len__"):  # has length
                    metadata["input_size"] = len(args[0])

            logger.debug(f"Starting {op_name}", **metadata)

            # Execute function with timing
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Add result metadata
                if include_result:
                    if hasattr(result, "shape"):  # numpy array
                        metadata["output_shape"] = str(result.shape)
                    elif hasattr(result, "__len__"):  # has length
                        metadata["output_size"] = len(result)
                    metadata["result_type"] = type(result).__name__

                # Log performance
                logger.log_performance(op_name, duration, **metadata)

                return result

            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Error in {op_name}: {str(e)}",
                    operation=op_name,
                    duration=duration,
                    error=str(e),
                )
                raise

        return wrapper

    return decorator


# Global cache for lightweight performance monitoring
_performance_cache: Dict[str, float] = {}


@contextmanager
def performance_context(operation_name: str, **metadata: Any):
    """Context manager for monitoring operation performance.

    Args:
        operation_name: Name of the operation
        **metadata: Additional metadata to log

    Example:
        with performance_context("matrix_multiplication", size="512x512"):
            result = expensive_operation()
    """
    logger = get_logger()
    start_time = time.time()

    logger.debug(f"Starting {operation_name}", **metadata)

    try:
        yield
        duration = time.time() - start_time
        logger.log_performance(operation_name, duration, **metadata)
    except Exception as e:
        duration = time.time() - start_time
        logger.error(
            f"Error in {operation_name}: {str(e)}",
            operation=operation_name,
            duration=duration,
            error=str(e),
        )
        raise


class PerformanceProfiler:
    """Advanced performance profiling utilities."""

    def __init__(self, name: str = "profiler") -> None:
        """Initialize profiler.

        Args:
            name: Profiler name
        """
        self.name = name
        self.logger = get_logger()
        self._active_operations: Dict[str, float] = {}

    def start_operation(self, operation_id: str) -> None:
        """Start timing an operation.

        Args:
            operation_id: Unique identifier for the operation
        """
        self._active_operations[operation_id] = time.time()
        self.logger.debug(f"Started operation: {operation_id}")

    def end_operation(self, operation_id: str, **metadata: Any) -> float:
        """End timing an operation and log results.

        Args:
            operation_id: Operation identifier
            **metadata: Additional metadata

        Returns:
            Duration in seconds
        """
        if operation_id not in self._active_operations:
            self.logger.warning(f"Operation {operation_id} was not started")
            return 0.0

        duration = time.time() - self._active_operations.pop(operation_id)
        self.logger.log_performance(operation_id, duration, **metadata)
        return duration

    def profile_memory_usage(self) -> Dict[str, Any]:
        """Profile current memory usage (if psutil available).

        Returns:
            Memory usage information
        """
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            usage = {
                "rss": memory_info.rss,  # Resident Set Size
                "vms": memory_info.vms,  # Virtual Memory Size
                "percent": process.memory_percent(),
                "available": psutil.virtual_memory().available,
            }

            self.logger.debug("Memory usage", **usage)
            return usage

        except ImportError:
            self.logger.warning("psutil not available for memory profiling")
            return {}


# Convenience functions
def log_info(message: str, **kwargs: Any) -> None:
    """Log info message using global logger."""
    get_logger().info(message, **kwargs)


def log_error(message: str, **kwargs: Any) -> None:
    """Log error message using global logger."""
    get_logger().error(message, **kwargs)


def log_warning(message: str, **kwargs: Any) -> None:
    """Log warning message using global logger."""
    get_logger().warning(message, **kwargs)


def get_performance_metrics() -> Dict[str, Any]:
    """Get all performance metrics from global logger."""
    return get_logger().get_metrics()


def save_performance_metrics(filepath: str) -> None:
    """Save performance metrics to file."""
    get_logger().save_metrics(filepath)
