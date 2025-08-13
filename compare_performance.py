#!/usr/bin/env python3
"""Performance comparison script for HyperFlowX optimization results."""

import json
from datetime import datetime

def load_results(filename):
    """Load benchmark results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {filename} not found")
        return None

def compare_results():
    """Compare baseline and optimized benchmark results."""
    baseline = load_results('benchmark_results.json')
    optimized = load_results('optimized_benchmark_results.json')
    
    if not baseline or not optimized:
        print("Error: Could not load benchmark results")
        return
    
    print("🚀 HyperFlowX Performance Optimization Results")
    print("=" * 60)
    
    # Sorting comparison
    baseline_sort = baseline['results']['sorting']['speed_boost']
    optimized_sort = optimized['results']['sorting']['speed_boost']
    print(f"\n📊 Sorting Performance:")
    print(f"  Before: {baseline_sort:.2f}× faster than NumPy")
    print(f"  After:  {optimized_sort:.2f}× faster than NumPy") 
    print(f"  Improvement: {optimized_sort/baseline_sort:.2f}× better")
    
    # Matrix multiplication comparison
    baseline_matrix = baseline['results']['matrix_mult']['speed_boost']
    optimized_matrix = optimized['results']['matrix_mult']['speed_boost']
    print(f"\n📊 Matrix Multiplication Performance:")
    print(f"  Before: {baseline_matrix:.2f}× faster than NumPy")
    print(f"  After:  {optimized_matrix:.2f}× faster than NumPy")
    print(f"  Improvement: {optimized_matrix/baseline_matrix:.2f}× better")
    
    # Hashing comparison
    baseline_hash = baseline['results']['hashing']['speed_boost']
    optimized_hash = optimized['results']['hashing']['speed_boost']
    print(f"\n📊 Hashing Performance:")
    print(f"  Before: {baseline_hash:.2f}× faster than SHA-256")
    print(f"  After:  {optimized_hash:.2f}× faster than SHA-256")
    print(f"  Improvement: {optimized_hash/baseline_hash:.2f}× better")
    
    # ML comparison (time-based)
    baseline_ml = baseline['results']['ml']['hyperflowx_ai']['time']
    optimized_ml = optimized['results']['ml']['hyperflowx_ai']['time']
    print(f"\n📊 Machine Learning Performance:")
    print(f"  Before: {baseline_ml:.4f} sec")
    print(f"  After:  {optimized_ml:.4f} sec")
    print(f"  Improvement: {baseline_ml/optimized_ml:.2f}× faster")
    
    print(f"\n🎉 Summary: Successfully optimized HyperFlowX performance!")
    print(f"   Timestamps: {baseline['timestamp']} → {optimized['timestamp']}")

if __name__ == "__main__":
    compare_results()