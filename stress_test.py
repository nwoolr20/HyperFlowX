#!/usr/bin/env python3
"""
🚀 HyperFlowX Comprehensive Stress Testing Suite
===============================================

This module performs intensive stress testing of HyperFlowX algorithms
to validate performance claims and ensure correctness under extreme conditions.
"""

import time
import numpy as np
import gc
import psutil
import os
import traceback
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json

from hyperflowx.sorting import hybrid_sort, adaptive_sort
from hyperflowx.optimizations import adaptive_matrix_mult, fast_matrix_mult
from hyperflowx.security import pascal_diamond_hash
# Skip ML tests for now due to torch dependency issues
# from hyperflowx.ml_model import train_hyperflowx


class StressTestSuite:
    """Comprehensive stress testing suite for HyperFlowX."""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'sorting_tests': {},
            'matrix_tests': {},
            'hashing_tests': {},
            'ml_tests': {},
            'concurrency_tests': {},
            'memory_tests': {},
            'correctness_tests': {},
            'errors': []
        }
    
    def _get_system_info(self):
        """Collect system information for test context."""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}"
        }
    
    def _measure_memory_usage(self):
        """Get current memory usage in MB."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def _verify_correctness(self, result, expected, test_name, tolerance=1e-5):
        """Verify result correctness against expected values."""
        try:
            if isinstance(result, np.ndarray) and isinstance(expected, np.ndarray):
                if not np.allclose(result, expected, rtol=tolerance, atol=tolerance):
                    error_msg = f"❌ {test_name}: Results differ from expected values"
                    self.results['errors'].append(error_msg)
                    return False
            elif isinstance(result, (list, tuple)) and isinstance(expected, (list, tuple)):
                if not np.allclose(result, expected, rtol=tolerance, atol=tolerance):
                    error_msg = f"❌ {test_name}: Results differ from expected values"
                    self.results['errors'].append(error_msg)
                    return False
            else:
                if result != expected:
                    error_msg = f"❌ {test_name}: Results differ from expected values"
                    self.results['errors'].append(error_msg)
                    return False
            return True
        except Exception as e:
            error_msg = f"❌ {test_name}: Correctness check failed - {str(e)}"
            self.results['errors'].append(error_msg)
            return False

    def stress_test_sorting(self):
        """Comprehensive sorting stress tests."""
        print("🔥 SORTING STRESS TESTS")
        print("=" * 50)
        
        test_cases = [
            # (size, description)
            (1000, "Small arrays"),
            (10000, "Medium arrays"), 
            (100000, "Large arrays"),
            (1000000, "Very large arrays"),
            (5000000, "Extreme arrays"),
        ]
        
        data_patterns = [
            ('random', lambda size: np.random.randint(0, size, size)),
            ('sorted', lambda size: np.arange(size)),
            ('reverse_sorted', lambda size: np.arange(size)[::-1]),
            ('duplicates', lambda size: np.random.choice(100, size)),
            ('nearly_sorted', lambda size: self._create_nearly_sorted(size)),
            ('few_unique', lambda size: np.random.choice(10, size)),
        ]
        
        for size, size_desc in test_cases:
            print(f"\n📊 Testing {size_desc} ({size:,} elements)")
            
            for pattern_name, pattern_func in data_patterns:
                try:
                    # Create test data
                    test_data = pattern_func(size)
                    
                    # Memory before
                    mem_before = self._measure_memory_usage()
                    
                    # Test HyperFlowX sorting
                    start_time = time.time()
                    hfx_result = hybrid_sort(test_data.copy())
                    hfx_time = time.time() - start_time
                    
                    # Test NumPy sorting (reference)
                    start_time = time.time()
                    np_result = np.sort(test_data.copy())
                    np_time = time.time() - start_time
                    
                    # Memory after
                    mem_after = self._measure_memory_usage()
                    
                    # Verify correctness
                    is_correct = self._verify_correctness(
                        hfx_result, np_result, 
                        f"Sorting {pattern_name} {size}"
                    )
                    
                    # Store results
                    test_key = f"{size}_{pattern_name}"
                    self.results['sorting_tests'][test_key] = {
                        'size': size,
                        'pattern': pattern_name,
                        'hyperflowx_time': hfx_time,
                        'numpy_time': np_time,
                        'speed_ratio': np_time / hfx_time if hfx_time > 0 else 0,
                        'memory_delta_mb': mem_after - mem_before,
                        'is_correct': is_correct
                    }
                    
                    # Performance indicator
                    if hfx_time > 0:
                        speedup = np_time / hfx_time
                        if speedup > 1.0:
                            indicator = f"✅ {speedup:.2f}× FASTER"
                        else:
                            indicator = f"⚠️  {1/speedup:.2f}× slower"
                    else:
                        indicator = "⚠️  Too fast to measure"
                    
                    print(f"  {pattern_name:15s}: HFX={hfx_time:.4f}s, NP={np_time:.4f}s {indicator}")
                    
                    # Force garbage collection
                    del test_data, hfx_result, np_result
                    gc.collect()
                    
                except Exception as e:
                    error_msg = f"❌ Sorting test failed for {size} {pattern_name}: {str(e)}"
                    print(f"  {error_msg}")
                    self.results['errors'].append(error_msg)
                    traceback.print_exc()

    def _create_nearly_sorted(self, size):
        """Create a nearly sorted array with some out-of-order elements."""
        arr = np.arange(size)
        # Shuffle 5% of elements
        n_shuffle = max(1, size // 20)
        indices = np.random.choice(size, n_shuffle, replace=False)
        np.random.shuffle(arr[indices])
        return arr

    def stress_test_matrix_multiplication(self):
        """Comprehensive matrix multiplication stress tests."""
        print("\n🔥 MATRIX MULTIPLICATION STRESS TESTS")
        print("=" * 50)
        
        matrix_sizes = [
            (64, 64, "Small matrices"),
            (256, 256, "Medium matrices"),
            (512, 512, "Large matrices"),
            (1024, 1024, "Very large matrices"),
            (2048, 2048, "Extreme matrices"),
            (100, 2000, "Rectangular matrices"),
            (2000, 100, "Tall matrices"),
        ]
        
        for rows, cols, desc in matrix_sizes:
            try:
                print(f"\n📊 Testing {desc} ({rows}×{cols})")
                
                # Create test matrices
                A = np.random.randn(rows, cols).astype(np.float64)
                B = np.random.randn(cols, rows).astype(np.float64)
                
                # Memory before
                mem_before = self._measure_memory_usage()
                
                # Test HyperFlowX matrix multiplication
                start_time = time.time()
                hfx_result = adaptive_matrix_mult(A, B)
                hfx_time = time.time() - start_time
                
                # Test NumPy matrix multiplication
                start_time = time.time()
                np_result = np.dot(A, B)
                np_time = time.time() - start_time
                
                # Memory after
                mem_after = self._measure_memory_usage()
                
                # Verify correctness
                is_correct = self._verify_correctness(
                    hfx_result, np_result,
                    f"Matrix mult {rows}×{cols}",
                    tolerance=1e-10
                )
                
                # Store results
                test_key = f"{rows}x{cols}"
                self.results['matrix_tests'][test_key] = {
                    'shape': (rows, cols),
                    'hyperflowx_time': hfx_time,
                    'numpy_time': np_time,
                    'speed_ratio': np_time / hfx_time if hfx_time > 0 else 0,
                    'memory_delta_mb': mem_after - mem_before,
                    'is_correct': is_correct
                }
                
                # Performance indicator
                if hfx_time > 0:
                    speedup = np_time / hfx_time
                    if speedup > 1.0:
                        indicator = f"✅ {speedup:.2f}× FASTER"
                    else:
                        indicator = f"⚠️  {1/speedup:.2f}× slower"
                else:
                    indicator = "⚠️  Too fast to measure"
                
                print(f"  HFX: {hfx_time:.4f}s, NumPy: {np_time:.4f}s {indicator}")
                print(f"  Memory usage: {mem_after - mem_before:.1f} MB")
                
                # Force garbage collection
                del A, B, hfx_result, np_result
                gc.collect()
                
            except Exception as e:
                error_msg = f"❌ Matrix test failed for {rows}×{cols}: {str(e)}"
                print(f"  {error_msg}")
                self.results['errors'].append(error_msg)
                traceback.print_exc()

    def stress_test_hashing(self):
        """Comprehensive hashing stress tests."""
        print("\n🔥 HASHING STRESS TESTS")
        print("=" * 50)
        
        data_sizes = [
            (256, "Small data"),
            (1024, "Medium data"),
            (10240, "Large data"), 
            (102400, "Very large data"),
            (1048576, "Extreme data (1MB)"),
        ]
        
        for size, desc in data_sizes:
            try:
                print(f"\n📊 Testing {desc} ({size} bytes)")
                
                # Create test data
                test_data = np.random.bytes(size)
                
                # Memory before
                mem_before = self._measure_memory_usage()
                
                # Test HyperFlowX hashing
                start_time = time.time()
                hfx_result = pascal_diamond_hash(test_data)
                hfx_time = time.time() - start_time
                
                # Test SHA-256 (reference)
                import hashlib
                start_time = time.time()
                sha_result = hashlib.sha256(test_data).hexdigest()
                sha_time = time.time() - start_time
                
                # Memory after
                mem_after = self._measure_memory_usage()
                
                # Store results
                self.results['hashing_tests'][str(size)] = {
                    'size_bytes': size,
                    'hyperflowx_time': hfx_time,
                    'sha256_time': sha_time,
                    'speed_ratio': sha_time / hfx_time if hfx_time > 0 else 0,
                    'memory_delta_mb': mem_after - mem_before,
                    'output_length': len(hfx_result)
                }
                
                # Performance indicator
                if hfx_time > 0:
                    speedup = sha_time / hfx_time
                    if speedup > 1.0:
                        indicator = f"✅ {speedup:.2f}× FASTER than SHA-256"
                    else:
                        indicator = f"⚠️  {1/speedup:.2f}× slower than SHA-256"
                else:
                    indicator = "⚠️  Too fast to measure"
                
                print(f"  HFX: {hfx_time:.6f}s, SHA-256: {sha_time:.6f}s {indicator}")
                
                # Force garbage collection
                del test_data
                gc.collect()
                
            except Exception as e:
                error_msg = f"❌ Hashing test failed for {size} bytes: {str(e)}"
                print(f"  {error_msg}")
                self.results['errors'].append(error_msg)
                traceback.print_exc()

    def stress_test_ml_training(self):
        """Stress test ML model training."""
        print("\n🔥 ML TRAINING STRESS TESTS")
        print("=" * 50)
        print("⚠️  Skipping ML tests due to torch dependency issues")
        
        # Skip ML tests for now
        self.results['ml_tests']['skipped'] = {
            'reason': 'torch dependency not available',
            'timestamp': datetime.now().isoformat()
        }

    def stress_test_concurrency(self):
        """Test concurrent operations."""
        print("\n🔥 CONCURRENCY STRESS TESTS")
        print("=" * 50)
        
        def concurrent_sorting_task():
            """Concurrent sorting task."""
            arr = np.random.randint(0, 10000, 10000)
            return hybrid_sort(arr)
        
        def concurrent_matrix_task():
            """Concurrent matrix multiplication task."""
            A = np.random.randn(256, 256)
            B = np.random.randn(256, 256)
            return adaptive_matrix_mult(A, B)
        
        def concurrent_hash_task():
            """Concurrent hashing task."""
            data = np.random.bytes(1024)
            return pascal_diamond_hash(data)
        
        # Test concurrent operations
        n_threads = min(8, psutil.cpu_count())
        n_tasks = 20
        
        try:
            print(f"📊 Testing {n_tasks} concurrent operations with {n_threads} threads")
            
            # Memory before
            mem_before = self._measure_memory_usage()
            
            # Run concurrent sorting
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(concurrent_sorting_task) for _ in range(n_tasks)]
                sorting_results = [f.result() for f in as_completed(futures)]
            sorting_time = time.time() - start_time
            
            # Run concurrent matrix operations
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(concurrent_matrix_task) for _ in range(n_tasks)]
                matrix_results = [f.result() for f in as_completed(futures)]
            matrix_time = time.time() - start_time
            
            # Run concurrent hashing
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                futures = [executor.submit(concurrent_hash_task) for _ in range(n_tasks)]
                hash_results = [f.result() for f in as_completed(futures)]
            hash_time = time.time() - start_time
            
            # Memory after
            mem_after = self._measure_memory_usage()
            
            # Store results
            self.results['concurrency_tests'] = {
                'n_threads': n_threads,
                'n_tasks': n_tasks,
                'sorting_time': sorting_time,
                'matrix_time': matrix_time,
                'hashing_time': hash_time,
                'memory_delta_mb': mem_after - mem_before,
                'sorting_results_count': len(sorting_results),
                'matrix_results_count': len(matrix_results),
                'hash_results_count': len(hash_results)
            }
            
            print(f"  Concurrent sorting: {sorting_time:.4f}s ({len(sorting_results)} completed)")
            print(f"  Concurrent matrix ops: {matrix_time:.4f}s ({len(matrix_results)} completed)")
            print(f"  Concurrent hashing: {hash_time:.4f}s ({len(hash_results)} completed)")
            print(f"  Memory usage: {mem_after - mem_before:.1f} MB")
            
            # Force garbage collection
            del sorting_results, matrix_results, hash_results
            gc.collect()
            
        except Exception as e:
            error_msg = f"❌ Concurrency test failed: {str(e)}"
            print(f"  {error_msg}")
            self.results['errors'].append(error_msg)
            traceback.print_exc()

    def stress_test_memory_pressure(self):
        """Test operations under memory pressure."""
        print("\n🔥 MEMORY PRESSURE STRESS TESTS")
        print("=" * 50)
        
        try:
            # Create memory pressure by allocating large arrays
            print("📊 Testing under memory pressure")
            
            initial_memory = self._measure_memory_usage()
            
            # Allocate large arrays to create memory pressure
            memory_hogs = []
            for i in range(5):
                memory_hogs.append(np.random.randn(1000000))  # ~8MB each
            
            pressure_memory = self._measure_memory_usage()
            
            # Test sorting under pressure
            test_array = np.random.randint(0, 100000, 100000)
            start_time = time.time()
            sorted_result = hybrid_sort(test_array)
            sorting_time = time.time() - start_time
            
            # Test matrix multiplication under pressure
            A = np.random.randn(512, 512)
            B = np.random.randn(512, 512)
            start_time = time.time()
            matrix_result = adaptive_matrix_mult(A, B)
            matrix_time = time.time() - start_time
            
            final_memory = self._measure_memory_usage()
            
            # Store results
            self.results['memory_tests'] = {
                'initial_memory_mb': initial_memory,
                'pressure_memory_mb': pressure_memory,
                'final_memory_mb': final_memory,
                'memory_pressure_mb': pressure_memory - initial_memory,
                'sorting_time_under_pressure': sorting_time,
                'matrix_time_under_pressure': matrix_time,
                'operations_completed': True
            }
            
            print(f"  Initial memory: {initial_memory:.1f} MB")
            print(f"  Under pressure: {pressure_memory:.1f} MB (+{pressure_memory - initial_memory:.1f} MB)")
            print(f"  Final memory: {final_memory:.1f} MB")
            print(f"  Sorting under pressure: {sorting_time:.4f}s")
            print(f"  Matrix ops under pressure: {matrix_time:.4f}s")
            print("  ✅ All operations completed successfully under memory pressure")
            
            # Clean up
            del memory_hogs, test_array, sorted_result, A, B, matrix_result
            gc.collect()
            
        except Exception as e:
            error_msg = f"❌ Memory pressure test failed: {str(e)}"
            print(f"  {error_msg}")
            self.results['errors'].append(error_msg)
            traceback.print_exc()

    def run_all_stress_tests(self):
        """Run all stress tests."""
        print("🚀 HYPERFLOWX COMPREHENSIVE STRESS TEST SUITE")
        print("=" * 60)
        print(f"Started at: {self.results['timestamp']}")
        print(f"System: {self.results['system_info']['cpu_count']} CPUs, {self.results['system_info']['memory_gb']:.1f} GB RAM")
        print("=" * 60)
        
        # Run all test categories
        self.stress_test_sorting()
        self.stress_test_matrix_multiplication()  
        self.stress_test_hashing()
        self.stress_test_ml_training()
        self.stress_test_concurrency()
        self.stress_test_memory_pressure()
        
        # Generate summary report
        self._generate_summary_report()
        
        # Save results
        self._save_results()
        
        return self.results

    def _generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "=" * 60)
        print("🏁 STRESS TEST SUMMARY REPORT")
        print("=" * 60)
        
        total_tests = (len(self.results['sorting_tests']) + 
                      len(self.results['matrix_tests']) + 
                      len(self.results['hashing_tests']) +
                      len(self.results['ml_tests']) + 
                      (1 if self.results['concurrency_tests'] else 0) +
                      (1 if self.results['memory_tests'] else 0))
        
        error_count = len(self.results['errors'])
        success_rate = ((total_tests - error_count) / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 OVERALL RESULTS:")
        print(f"   Total tests run: {total_tests}")
        print(f"   Successful tests: {total_tests - error_count}")
        print(f"   Failed tests: {error_count}")
        print(f"   Success rate: {success_rate:.1f}%")
        
        # Sorting performance summary
        if self.results['sorting_tests']:
            print(f"\n🔄 SORTING PERFORMANCE:")
            sorting_speedups = [t['speed_ratio'] for t in self.results['sorting_tests'].values() if t['is_correct'] and t['speed_ratio'] > 0]
            if sorting_speedups:
                avg_speedup = np.mean(sorting_speedups)
                max_speedup = np.max(sorting_speedups)
                min_speedup = np.min(sorting_speedups)
                print(f"   Average speedup vs NumPy: {avg_speedup:.2f}×")
                print(f"   Best speedup: {max_speedup:.2f}×")
                print(f"   Worst speedup: {min_speedup:.2f}×")
        
        # Matrix performance summary
        if self.results['matrix_tests']:
            print(f"\n🔢 MATRIX MULTIPLICATION PERFORMANCE:")
            matrix_speedups = [t['speed_ratio'] for t in self.results['matrix_tests'].values() if t['is_correct'] and t['speed_ratio'] > 0]
            if matrix_speedups:
                avg_speedup = np.mean(matrix_speedups)
                max_speedup = np.max(matrix_speedups)
                min_speedup = np.min(matrix_speedups)
                print(f"   Average speedup vs NumPy: {avg_speedup:.2f}×")
                print(f"   Best speedup: {max_speedup:.2f}×")
                print(f"   Worst speedup: {min_speedup:.2f}×")
        
        # Hashing performance summary
        if self.results['hashing_tests']:
            print(f"\n🔐 HASHING PERFORMANCE:")
            hash_speedups = [t['speed_ratio'] for t in self.results['hashing_tests'].values() if t['speed_ratio'] > 0]
            if hash_speedups:
                avg_speedup = np.mean(hash_speedups)
                max_speedup = np.max(hash_speedups)
                min_speedup = np.min(hash_speedups)
                print(f"   Average speedup vs SHA-256: {avg_speedup:.2f}×")
                print(f"   Best speedup: {max_speedup:.2f}×")
                print(f"   Worst speedup: {min_speedup:.2f}×")
        
        # Error summary
        if error_count > 0:
            print(f"\n❌ ERRORS ENCOUNTERED:")
            for error in self.results['errors'][:5]:  # Show first 5 errors
                print(f"   • {error}")
            if error_count > 5:
                print(f"   ... and {error_count - 5} more errors")
        else:
            print(f"\n✅ ALL TESTS PASSED - NO ERRORS DETECTED!")
        
        print("\n" + "=" * 60)

    def _save_results(self, filename="stress_test_results.json"):
        """Save stress test results to file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            print(f"📄 Detailed results saved to {filename}")
        except Exception as e:
            print(f"❌ Failed to save results: {e}")


def main():
    """Run the comprehensive stress test suite."""
    print("🚀 Initializing HyperFlowX Stress Test Suite...")
    
    # Run stress tests
    suite = StressTestSuite()
    results = suite.run_all_stress_tests()
    
    # Final memory cleanup
    gc.collect()
    
    print("\n🏁 Stress testing completed!")
    return results


if __name__ == "__main__":
    main()