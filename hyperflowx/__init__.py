# hyperflowx/__init__.py
"""HyperFlowX: High-Performance Computing Library with AI-Powered Optimizations."""

from typing import Any
from .__version__ import __version__, __author__, __email__, __license__, __description__, __url__
from .sorting import hybrid_sort
from .ai_automation import automate_hyperflowx_fixes


def train_hyperflowx(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and invoke :func:`ml_model.train_hyperflowx`."""
    from .ml_model import train_hyperflowx as _train
    return _train(*args, **kwargs)


def fast_matrix_mult(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and invoke :func:`optimizations.adaptive_matrix_mult`."""
    from .optimizations import adaptive_matrix_mult as _fast
    return _fast(*args, **kwargs)


def async_pipeline(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and invoke :func:`pipeline.async_pipeline`."""
    from .pipeline import async_pipeline as _pipeline
    return _pipeline(*args, **kwargs)


def pascal_diamond_hash(*args: Any, **kwargs: Any) -> Any:
    """Lazily import and invoke :func:`security.pascal_diamond_hash`."""
    from .security import pascal_diamond_hash as _hash
    return _hash(*args, **kwargs)

__all__ = [
    "hybrid_sort",
    "train_hyperflowx",
    "fast_matrix_mult",
    "async_pipeline",
    "pascal_diamond_hash",
    "automate_hyperflowx_fixes"  # ✅ Now available for easy import
]
