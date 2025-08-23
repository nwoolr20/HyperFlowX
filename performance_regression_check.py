#!/usr/bin/env python3
"""Performance regression detection for HyperFlowX CI/CD pipeline.

This script runs benchmarks and compares performance against historical baselines
to detect performance regressions automatically.
"""

import json
import os
import sys
import time
import subprocess
from typing import Dict, Any, List, Tuple
from pathlib import Path
import tempfile
import numpy as np

# Import HyperFlowX modules for benchmarking
from hyperflowx.sorting import hybrid_sort
from hyperflowx.optimizations import adaptive_matrix_mult
from hyperflowx.security import pascal_diamond_hash
from hyperflowx.monitoring import configure_logging, get_performance_metrics


class PerformanceRegessionDetector:
    """Detects performance regressions by comparing against baselines."""
    
    def __init__(self, baseline_file: str = "performance_baseline.json") -> None:
        """Initialize regression detector.
        
        Args:
            baseline_file: Path to baseline performance data
        """
        self.baseline_file = baseline_file
        self.current_metrics: Dict[str, Any] = {}
        self.baseline_metrics: Dict[str, Any] = {}
        self.regression_threshold = 0.15  # 15% regression threshold
        self.improvement_threshold = 0.10  # 10% improvement to update baseline
        
        # Configure minimal logging for CI
        configure_logging(level="WARNING")
    
    def load_baseline(self) -> None:
        """Load baseline performance metrics."""
        if os.path.exists(self.baseline_file):
            try:
                with open(self.baseline_file, 'r') as f:
                    self.baseline_metrics = json.load(f)
                print(f"✅ Loaded baseline from {self.baseline_file}")
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Could not load baseline: {e}")
                self.baseline_metrics = {}
        else:
            print(f"⚠️  No baseline file found at {self.baseline_file}")
            self.baseline_metrics = {}
    
    def save_baseline(self) -> None:
        """Save current metrics as new baseline."""
        try:
            with open(self.baseline_file, 'w') as f:
                json.dump(self.current_metrics, f, indent=2)
            print(f"✅ Saved new baseline to {self.baseline_file}")
        except IOError as e:
            print(f"❌ Could not save baseline: {e}")
    
    def run_performance_tests(self) -> None:
        """Run standardized performance tests."""
        print("🚀 Running performance benchmark suite...")
        
        # Sorting benchmark
        print("  📊 Testing sorting performance...")
        data_sizes = [1000, 10000, 50000]
        
        for size in data_sizes:
            test_data = np.random.randint(0, size, size)
            
            # Time multiple runs for statistical significance
            times = []
            for _ in range(5):
                start = time.time()
                hybrid_sort(test_data.copy())
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            self.current_metrics[f"sorting_{size}"] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "throughput": size / avg_time  # elements per second
            }
        
        # Matrix multiplication benchmark
        print("  🔢 Testing matrix multiplication performance...")
        matrix_sizes = [128, 256, 512]
        
        for size in matrix_sizes:
            A = np.random.rand(size, size)
            B = np.random.rand(size, size)
            
            times = []
            for _ in range(3):  # Fewer runs for large matrices
                start = time.time()
                adaptive_matrix_mult(A, B)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            self.current_metrics[f"matrix_mult_{size}x{size}"] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "throughput": (size * size * size) / avg_time  # operations per second
            }
        
        # Hashing benchmark
        print("  🔐 Testing hashing performance...")
        data_sizes = [256, 1024, 4096]
        
        for size in data_sizes:
            test_data = np.random.bytes(size)
            
            times = []
            for _ in range(10):
                start = time.time()
                pascal_diamond_hash(test_data)
                times.append(time.time() - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            self.current_metrics[f"hashing_{size}"] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "throughput": size / avg_time  # bytes per second
            }
        
        print("✅ Performance benchmarks completed")
    
    def compare_performance(self) -> Tuple[List[str], List[str], List[str]]:
        """Compare current performance against baseline.
        
        Returns:
            Tuple of (regressions, improvements, new_tests)
        """
        regressions = []
        improvements = []
        new_tests = []
        
        for test_name, current_data in self.current_metrics.items():
            if test_name not in self.baseline_metrics:
                new_tests.append(test_name)
                continue
            
            baseline_data = self.baseline_metrics[test_name]
            current_time = current_data["avg_time"]
            baseline_time = baseline_data["avg_time"]
            
            # Calculate percentage change (positive = slower = regression)
            change_percent = (current_time - baseline_time) / baseline_time
            
            if change_percent > self.regression_threshold:
                regression_msg = (
                    f"{test_name}: {change_percent:.1%} slower "
                    f"({current_time:.4f}s vs {baseline_time:.4f}s baseline)"
                )
                regressions.append(regression_msg)
            
            elif change_percent < -self.improvement_threshold:
                improvement_msg = (
                    f"{test_name}: {-change_percent:.1%} faster "
                    f"({current_time:.4f}s vs {baseline_time:.4f}s baseline)"
                )
                improvements.append(improvement_msg)
        
        return regressions, improvements, new_tests
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        regressions, improvements, new_tests = self.compare_performance()
        
        report = {
            "timestamp": time.time(),
            "summary": {
                "total_tests": len(self.current_metrics),
                "regressions": len(regressions),
                "improvements": len(improvements),
                "new_tests": len(new_tests)
            },
            "regressions": regressions,
            "improvements": improvements,
            "new_tests": new_tests,
            "current_metrics": self.current_metrics,
            "baseline_available": bool(self.baseline_metrics)
        }
        
        return report
    
    def print_report(self, report: Dict[str, Any]) -> None:
        """Print human-readable performance report."""
        print("\n" + "=" * 60)
        print("🎯 PERFORMANCE REGRESSION ANALYSIS")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"📈 Total tests: {summary['total_tests']}")
        print(f"🔴 Regressions: {summary['regressions']}")
        print(f"🟢 Improvements: {summary['improvements']}")
        print(f"🆕 New tests: {summary['new_tests']}")
        
        if report["regressions"]:
            print(f"\n❌ PERFORMANCE REGRESSIONS DETECTED:")
            for regression in report["regressions"]:
                print(f"  • {regression}")
        
        if report["improvements"]:
            print(f"\n✅ PERFORMANCE IMPROVEMENTS:")
            for improvement in report["improvements"]:
                print(f"  • {improvement}")
        
        if report["new_tests"]:
            print(f"\n🆕 NEW PERFORMANCE TESTS:")
            for test in report["new_tests"]:
                print(f"  • {test}")
        
        if not report["baseline_available"]:
            print(f"\n⚠️  No baseline available - this will be the new baseline")
        
        print("=" * 60 + "\n")


def main() -> int:
    """Main function for performance regression detection."""
    print("🔍 HyperFlowX Performance Regression Detection")
    print("=" * 50)
    
    # Get baseline file path from environment or use default
    baseline_file = os.environ.get("PERFORMANCE_BASELINE_FILE", "performance_baseline.json")
    
    detector = PerformanceRegessionDetector(baseline_file)
    
    # Load existing baseline
    detector.load_baseline()
    
    # Run performance tests
    detector.run_performance_tests()
    
    # Generate and print report
    report = detector.generate_report()
    detector.print_report(report)
    
    # Check if we should fail the build
    regressions = report["regressions"]
    
    if regressions:
        print("❌ BUILD FAILED: Performance regressions detected!")
        
        # In CI, optionally allow continuing if no baseline exists
        if not report["baseline_available"] and os.environ.get("CI"):
            print("⚠️  No baseline available in CI - saving current performance as baseline")
            detector.save_baseline()
            return 0
        
        return 1  # Exit with error code
    
    else:
        print("✅ BUILD PASSED: No performance regressions detected!")
        
        # Update baseline if we have significant improvements or no baseline
        improvements = report["improvements"]
        if improvements or not report["baseline_available"]:
            print("📝 Updating performance baseline with current results")
            detector.save_baseline()
        
        return 0


if __name__ == "__main__":
    sys.exit(main())