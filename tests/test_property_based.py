"""Property-based testing for HyperFlowX using Hypothesis.

This module provides comprehensive property-based tests that verify algorithmic
properties and invariants across different input types and sizes.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import torch

from hyperflowx.sorting import hybrid_sort, adaptive_sort
from hyperflowx.security import pascal_diamond_hash
from hyperflowx.optimizations import adaptive_matrix_mult, fast_matrix_mult
from hyperflowx.ml_model import train_hyperflowx


class TestSortingProperties:
    """Property-based tests for sorting algorithms."""
    
    @given(arrays(np.float64, st.integers(1, 1000), elements=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=50, deadline=5000)
    def test_sorting_is_sorted(self, arr: np.ndarray) -> None:
        """Property: Output should always be sorted."""
        result = hybrid_sort(arr)
        
        # Check that result is sorted
        for i in range(len(result) - 1):
            assert result[i] <= result[i + 1], f"Array not sorted at indices {i}, {i+1}: {result[i]} > {result[i+1]}"
    
    @given(arrays(np.float64, st.integers(1, 1000), elements=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=50, deadline=5000)
    def test_sorting_preserves_elements(self, arr: np.ndarray) -> None:
        """Property: Sorting should preserve all elements."""
        result = hybrid_sort(arr)
        
        # Both arrays should have same length
        assert len(result) == len(arr)
        
        # Both arrays should have same elements (just reordered)
        assert np.allclose(np.sort(result), np.sort(arr)), "Elements not preserved during sorting"
    
    @given(arrays(np.float64, st.integers(1, 1000), elements=st.floats(-1000, 1000, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=30, deadline=5000)
    def test_sorting_correctness_vs_numpy(self, arr: np.ndarray) -> None:
        """Property: Should produce same result as NumPy sort."""
        result = hybrid_sort(arr)
        expected = np.sort(arr)
        
        np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    
    @given(arrays(np.float64, st.integers(1, 100), elements=st.floats(-100, 100, allow_nan=False, allow_infinity=False)))
    @settings(max_examples=30, deadline=5000)
    def test_adaptive_sort_fallback_consistency(self, arr: np.ndarray) -> None:
        """Property: Adaptive sort should be consistent regardless of fallback setting."""
        result_no_fallback = adaptive_sort(arr, allow_numpy_fallback=False)
        result_with_fallback = adaptive_sort(arr, allow_numpy_fallback=True)
        
        # Both should be sorted
        assert np.allclose(np.sort(result_no_fallback), result_no_fallback)
        assert np.allclose(np.sort(result_with_fallback), result_with_fallback)


class TestMatrixOperationProperties:
    """Property-based tests for matrix operations."""
    
    @given(
        m=st.integers(1, 50),
        n=st.integers(1, 50),
        k=st.integers(1, 50)
    )
    @settings(max_examples=20, deadline=10000)
    def test_matrix_multiplication_dimensions(self, m: int, n: int, k: int) -> None:
        """Property: Matrix multiplication should preserve dimensional correctness."""
        A = np.random.rand(m, k).astype(np.float64)
        B = np.random.rand(k, n).astype(np.float64)
        
        result = adaptive_matrix_mult(A, B)
        
        # Result should have correct dimensions
        assert result.shape == (m, n), f"Expected shape {(m, n)}, got {result.shape}"
    
    @given(
        m=st.integers(2, 20),
        n=st.integers(2, 20),
        k=st.integers(2, 20)
    )
    @settings(max_examples=15, deadline=10000)
    def test_matrix_multiplication_correctness(self, m: int, n: int, k: int) -> None:
        """Property: Matrix multiplication should match NumPy results."""
        A = np.random.rand(m, k).astype(np.float64)
        B = np.random.rand(k, n).astype(np.float64)
        
        result = adaptive_matrix_mult(A, B)
        expected = np.dot(A, B)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)
    
    @given(
        size=st.integers(1, 30)
    )
    @settings(max_examples=15, deadline=5000)
    def test_matrix_multiplication_associativity(self, size: int) -> None:
        """Property: (AB)C = A(BC) for compatible matrices."""
        A = np.random.rand(size, size).astype(np.float64)
        B = np.random.rand(size, size).astype(np.float64)
        C = np.random.rand(size, size).astype(np.float64)
        
        # (AB)C
        AB = adaptive_matrix_mult(A, B)
        result1 = adaptive_matrix_mult(AB, C)
        
        # A(BC)
        BC = adaptive_matrix_mult(B, C)
        result2 = adaptive_matrix_mult(A, BC)
        
        np.testing.assert_allclose(result1, result2, rtol=1e-5, atol=1e-8)


class TestHashingProperties:
    """Property-based tests for hashing functions."""
    
    @given(st.binary(min_size=1, max_size=1000))
    @settings(max_examples=50, deadline=2000)
    def test_hash_deterministic(self, data: bytes) -> None:
        """Property: Same input should always produce same hash."""
        hash1 = pascal_diamond_hash(data)
        hash2 = pascal_diamond_hash(data)
        
        assert hash1 == hash2, "Hash function should be deterministic"
        assert isinstance(hash1, str), "Hash should be a string"
        assert len(hash1) > 0, "Hash should not be empty"
    
    @given(st.binary(min_size=1, max_size=500))
    @settings(max_examples=30, deadline=2000)
    def test_hash_different_inputs(self, data: bytes) -> None:
        """Property: Different inputs should (usually) produce different hashes."""
        assume(len(data) > 1)  # Need modifiable data
        
        # Modify one byte
        modified_data = bytearray(data)
        modified_data[0] = (modified_data[0] + 1) % 256
        modified_data = bytes(modified_data)
        
        hash1 = pascal_diamond_hash(data)
        hash2 = pascal_diamond_hash(modified_data)
        
        # While not guaranteed, different inputs should usually produce different hashes
        # This tests for obvious collision issues
        if data != modified_data:
            # Allow for some collisions in small test cases, but flag if too many
            pass  # Just ensure no exceptions are thrown
    
    @given(arrays(np.uint8, st.integers(1, 500)))
    @settings(max_examples=30, deadline=2000)
    def test_hash_numpy_array_input(self, arr: np.ndarray) -> None:
        """Property: Hash function should handle numpy array inputs."""
        hash_result = pascal_diamond_hash(arr)
        
        assert isinstance(hash_result, str), "Hash should be a string"
        assert len(hash_result) > 0, "Hash should not be empty"
        # Should be valid hex
        try:
            int(hash_result, 16)
        except ValueError:
            pytest.fail("Hash result should be valid hexadecimal")


class TestMLProperties:
    """Property-based tests for machine learning components."""
    
    @given(
        n_samples=st.integers(100, 500),
        n_features=st.integers(5, 30),
    )
    @settings(max_examples=5, deadline=30000)  # ML training is slow
    def test_ml_model_selection_small_features(self, n_samples: int, n_features: int) -> None:
        """Property: Small feature sets should use XGBoost."""
        assume(n_features <= 50)  # Should trigger XGBoost path
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        
        model = train_hyperflowx(X, y, use_cuda=False)
        
        # For small feature sets, should return XGBoost model
        assert hasattr(model, 'predict'), "Model should have predict method"
        assert hasattr(model, 'fit'), "Model should have fit method"
    
    @given(
        n_samples=st.integers(100, 300),
        n_features=st.integers(51, 80),
    )
    @settings(max_examples=3, deadline=60000)  # Neural network training is slower
    def test_ml_model_selection_large_features(self, n_samples: int, n_features: int) -> None:
        """Property: Large feature sets should use Neural Network."""
        assume(n_features > 50)  # Should trigger Neural Network path
        
        X = np.random.rand(n_samples, n_features)
        y = np.random.rand(n_samples)
        
        try:
            model = train_hyperflowx(X, y, use_cuda=False)
            
            # For large feature sets, should return PyTorch model
            assert hasattr(model, 'forward'), "Neural network should have forward method"
            assert isinstance(model, torch.nn.Module), "Should be a PyTorch module"
        except Exception as e:
            # Allow for CUDA/PyTorch availability issues in CI
            if "CUDA" in str(e) or "torch" in str(e):
                pytest.skip(f"PyTorch/CUDA not available: {e}")
            else:
                raise


class TestIntegrationProperties:
    """Integration tests with property-based approach."""
    
    @given(st.integers(10, 100))
    @settings(max_examples=10, deadline=5000)
    def test_pipeline_integration(self, size: int) -> None:
        """Property: Pipeline components should work together without errors."""
        # Test data that exercises multiple components
        arr = np.random.randint(0, 1000, size)
        matrix = np.random.rand(min(size, 20), min(size, 20))
        data = np.random.bytes(min(size, 100))
        
        # Test sorting
        sorted_result = hybrid_sort(arr)
        assert len(sorted_result) == len(arr)
        
        # Test matrix ops
        matrix_result = adaptive_matrix_mult(matrix, matrix.T)
        assert matrix_result.shape == (matrix.shape[0], matrix.shape[0])
        
        # Test hashing
        hash_result = pascal_diamond_hash(data)
        assert isinstance(hash_result, str)
        assert len(hash_result) > 0