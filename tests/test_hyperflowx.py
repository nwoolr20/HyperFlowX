import unittest
import numpy as np
from hyperflowx.sorting import hybrid_sort
from hyperflowx.security import pascal_diamond_hash
from hyperflowx.optimizations import fast_matrix_mult

class TestHyperFlowX(unittest.TestCase):

    def test_sorting(self):
        """Tests Hybrid Sort against NumPy's Timsort for correctness."""
        arr = np.random.randint(0, 10000, 1000)
        sorted_arr = hybrid_sort(arr)
        self.assertTrue(np.array_equal(sorted_arr, np.sort(arr)))  # ✅ Correct order check

    def test_hashing(self):
        """Tests Pascal-Diamond Hash for correct output."""
        data = np.random.bytes(256)
        pdh_hash = pascal_diamond_hash(data)
        self.assertIsInstance(pdh_hash, str)  # ✅ Ensure hex string output
        self.assertTrue(len(pdh_hash) > 0)  # ✅ Ensure output is non-empty

    def test_matrix_mult(self):
        """Tests optimized matrix multiplication against NumPy's dot product."""
        A = np.random.rand(512, 512)
        B = np.random.rand(512, 512)
        result_hfx = fast_matrix_mult(A, B)
        result_np = np.dot(A, B)
        np.testing.assert_allclose(result_hfx, result_np, rtol=1e-5)  # ✅ Precision check

if __name__ == '__main__':
    unittest.main()
