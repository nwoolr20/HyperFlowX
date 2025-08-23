"""Setup configuration for HyperFlowX."""

from setuptools import setup, find_packages
import os

# Read version from __version__.py
def get_version():
    """Get version from __version__.py file."""
    version_file = os.path.join("hyperflowx", "__version__.py")
    with open(version_file, "r", encoding="utf-8") as f:
        exec(f.read())
    return locals()["__version__"]

# Read long description from README
def get_long_description():
    """Get long description from README.md."""
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return ""

# Core dependencies
INSTALL_REQUIRES = [
    "numpy>=2.1.3",
    "numba>=0.58.0",
    "scipy>=1.10.0",
    "xgboost>=1.7.0",
    "torch>=2.0.0",
    "scikit-learn>=1.3.0",
    "tqdm>=4.65.0",
]

# Development dependencies
DEV_REQUIRES = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "flake8>=6.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0",
    "isort>=5.12.0",
    "bandit>=1.7.0",
    "safety>=2.3.0",
    "pre-commit>=3.0.0",
]

# Documentation dependencies
DOCS_REQUIRES = [
    "sphinx>=5.0.0",
    "sphinx-rtd-theme>=1.2.0",
    "sphinx-autodoc-typehints>=1.20.0",
]

# Benchmark dependencies
BENCH_REQUIRES = [
    "matplotlib>=3.6.0",
    "seaborn>=0.12.0",
    "pandas>=1.5.0",
]

setup(
    name="HyperFlowX",
    version=get_version(),
    description="High-Performance Computing Library with AI-Powered Optimizations",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    author="HyperFlowX Team",
    author_email="team@hyperflowx.dev",
    url="https://github.com/nwoolr20/HyperFlowX",
    project_urls={
        "Bug Reports": "https://github.com/nwoolr20/HyperFlowX/issues",
        "Source": "https://github.com/nwoolr20/HyperFlowX",
        "Documentation": "https://github.com/nwoolr20/HyperFlowX#readme",
    },
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": DEV_REQUIRES,
        "docs": DOCS_REQUIRES,
        "bench": BENCH_REQUIRES,
        "all": DEV_REQUIRES + DOCS_REQUIRES + BENCH_REQUIRES,
    },
    entry_points={
        "console_scripts": [
            "hyperflowx=examples.run_hyperflowx:main",
            "hyperflowx-benchmark=hyperflowx.benchmark:main",
        ]
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11", 
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Distributed Computing",
    ],
    keywords=[
        "high-performance-computing",
        "machine-learning",
        "optimization",
        "algorithms",
        "parallel-computing",
        "gpu-acceleration",
        "numba",
        "pytorch",
        "workflow-orchestration",
        "data-pipeline",
    ],
    license="MIT",
    zip_safe=False,
)
