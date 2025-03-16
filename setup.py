from setuptools import setup, find_packages

setup(
    name="HyperFlowX",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=2.1.3",
        "numba",
        "scipy",
        "xgboost",
        "torch",
        "scikit-learn",
        "tqdm"
    ],
    entry_points={
        "console_scripts": [
            "hyperflowx=examples.run_hyperflowx:main"
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
