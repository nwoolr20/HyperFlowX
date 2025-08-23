# Contributing to HyperFlowX

We welcome contributions to HyperFlowX! This document provides guidelines for contributing to the project.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of high-performance computing concepts

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/yourusername/HyperFlowX.git
   cd HyperFlowX
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Development Dependencies**
   ```bash
   pip install -e ".[dev]"
   pip install pytest pytest-cov flake8 black mypy bandit safety
   ```

4. **Verify Installation**
   ```bash
   python -m pytest tests/
   python hyperflowx/benchmark.py
   ```

## 🔧 Development Workflow

### Branch Strategy

- `main`: Production-ready code
- `develop`: Integration branch for features
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes

### Making Changes

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow the coding standards below
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run tests
   python -m pytest tests/ -v
   
   # Run linting
   flake8 hyperflowx
   black --check hyperflowx
   mypy hyperflowx --ignore-missing-imports
   
   # Run benchmarks
   python hyperflowx/benchmark.py
   ```

4. **Commit Changes**
   ```bash
   git add .
   git commit -m "feat: add new optimization algorithm"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📋 Coding Standards

### Code Style

- **Formatting**: Use [Black](https://black.readthedocs.io/) with default settings
- **Line Length**: 88 characters (Black default)
- **Imports**: Group imports using [isort](https://pycqa.github.io/isort/)
- **Type Hints**: Add type hints for public APIs

### Code Quality

- **Linting**: Code must pass flake8 checks
- **Type Checking**: Use mypy for static type checking
- **Security**: Pass bandit security checks
- **Performance**: Benchmark critical paths

### Documentation

- **Docstrings**: Use Google-style docstrings
- **Comments**: Explain complex algorithms and optimizations
- **README**: Update if adding new features or changing APIs

### Example Function

```python
import numpy as np
from typing import Union, Tuple

def optimize_algorithm(
    data: np.ndarray, 
    threshold: float = 1.0
) -> Tuple[np.ndarray, float]:
    """Optimize algorithm performance using adaptive techniques.
    
    Args:
        data: Input data array to process
        threshold: Performance threshold for algorithm selection
        
    Returns:
        Tuple of (optimized_result, performance_metric)
        
    Raises:
        ValueError: If data is empty or threshold is negative
        
    Example:
        >>> data = np.random.rand(1000)
        >>> result, metric = optimize_algorithm(data, 0.5)
        >>> assert result.shape == data.shape
    """
    if len(data) == 0:
        raise ValueError("Data cannot be empty")
    if threshold < 0:
        raise ValueError("Threshold must be non-negative")
        
    # Implementation here
    result = data.copy()
    metric = 1.0
    
    return result, metric
```

## 🧪 Testing Guidelines

### Test Structure

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test module interactions
- **Performance Tests**: Benchmark critical paths
- **Property Tests**: Use hypothesis for property-based testing

### Test Requirements

- All new code must have tests
- Aim for >90% code coverage
- Tests must be deterministic and isolated
- Include edge cases and error conditions

### Test Example

```python
import pytest
import numpy as np
from hyperflowx.sorting import hybrid_sort

class TestHybridSort:
    def test_empty_array(self):
        """Test sorting empty array."""
        result = hybrid_sort(np.array([]))
        assert len(result) == 0
    
    def test_single_element(self):
        """Test sorting single element."""
        data = np.array([42])
        result = hybrid_sort(data)
        assert np.array_equal(result, data)
    
    def test_random_data(self):
        """Test sorting random data."""
        data = np.random.randint(0, 1000, 1000)
        result = hybrid_sort(data)
        expected = np.sort(data)
        assert np.array_equal(result, expected)
    
    @pytest.mark.performance
    def test_performance_large_array(self):
        """Test performance on large arrays."""
        data = np.random.randint(0, 100000, 100000)
        import time
        start = time.time()
        hybrid_sort(data)
        elapsed = time.time() - start
        assert elapsed < 5.0  # Should complete in under 5 seconds
```

## 🚦 Pull Request Process

### Before Submitting

1. **Ensure CI Passes**
   - All tests pass
   - Code coverage maintained
   - Linting passes
   - Security checks pass

2. **Update Documentation**
   - Update README if needed
   - Add docstrings to new functions
   - Update CHANGELOG.md

3. **Performance Impact**
   - Run benchmarks on affected code
   - Document any performance changes
   - Consider backward compatibility

### PR Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Performance improvement
- [ ] Documentation update

## Testing
- [ ] Tests added for new functionality
- [ ] All existing tests pass
- [ ] Benchmarks run and documented
- [ ] Manual testing completed

## Performance Impact
Describe any performance implications.

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new security vulnerabilities
```

## 🐛 Reporting Issues

### Bug Reports

When reporting bugs, include:

- **Environment**: Python version, OS, HyperFlowX version
- **Reproduction Steps**: Minimal code to reproduce the issue
- **Expected Behavior**: What should happen
- **Actual Behavior**: What actually happens
- **Error Messages**: Full traceback if applicable

### Feature Requests

For feature requests, include:

- **Use Case**: Why is this feature needed?
- **Proposed Solution**: How should it work?
- **Alternatives**: Other solutions considered
- **Performance Impact**: Expected impact on performance

## 📚 Resources

- [HyperFlowX Documentation](README.md)
- [NumPy Documentation](https://numpy.org/doc/)
- [Numba Documentation](https://numba.pydata.org/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Performance Best Practices](https://docs.python.org/3/howto/perf_profiling.html)

## 📞 Community

- **Discussions**: Use GitHub Discussions for questions
- **Issues**: GitHub Issues for bugs and feature requests
- **Discord**: Join our Discord for real-time discussions (if available)

## 📄 License

By contributing to HyperFlowX, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to HyperFlowX! Together we're building the future of high-performance computing. 🚀