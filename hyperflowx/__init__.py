# hyperflowx/__init__.py

from .sorting import hybrid_sort
from .ml_model import train_hyperflowx
from .optimizations import fast_matrix_mult
from .pipeline import async_pipeline
from .security import pascal_diamond_hash  # ✅ Added missing import
from .ai_automation import automate_hyperflowx_fixes  # ✅ Added AI-powered automation

__all__ = [
    "hybrid_sort",
    "train_hyperflowx",
    "fast_matrix_mult",
    "async_pipeline",
    "pascal_diamond_hash",
    "automate_hyperflowx_fixes"  # ✅ Now available for easy import
]
