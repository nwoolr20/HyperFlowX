---
name: Performance Issue
about: Report performance problems or regressions
title: '[PERFORMANCE] '
labels: ['performance', 'needs-benchmark']
assignees: ''

---

## ⚡ Performance Issue

### Performance Problem
A clear description of the performance issue you're experiencing.

### Environment
- **HyperFlowX Version**: [e.g., 2.0.0]
- **Python Version**: [e.g., 3.11.0]
- **Operating System**: [e.g., Ubuntu 22.04]
- **CPU**: [e.g., Intel i7-12700K, AMD Ryzen 7 5800X]
- **RAM**: [e.g., 32GB DDR4]
- **GPU**: [e.g., NVIDIA RTX 3080, AMD RX 6800 XT, None]
- **CUDA Version**: [if using GPU acceleration]

### Benchmark Results
Please provide benchmark data:

#### Current Performance
```
# Paste benchmark output here
HyperFlowX Sort: X.XXX sec
HyperFlowX Matrix Mult: X.XXX sec
# etc.
```

#### Expected Performance
- **Baseline**: [What performance did you expect?]
- **Reference**: [Compared to NumPy, previous version, etc.]
- **Target**: [What would be acceptable performance?]

### Reproduction
```python
# Minimal code to reproduce the performance issue
import hyperflowx
import time

# Your performance test code here
start = time.time()
result = hyperflowx.some_function(data)
print(f"Time: {time.time() - start:.4f} sec")
```

### Data Characteristics
- **Data Size**: [e.g., 1M elements, 512x512 matrix]
- **Data Type**: [e.g., float64, int32]
- **Data Pattern**: [e.g., random, sorted, sparse]
- **Memory Usage**: [Peak memory consumption]

### Performance Analysis
If you've done any profiling:
- **Bottlenecks**: [Where is time being spent?]
- **CPU Usage**: [Single-core, multi-core utilization]
- **Memory Access**: [Cache misses, memory bandwidth]
- **Compiler**: [JIT compilation overhead, optimization flags]

### Comparison
How does performance compare to:
- [ ] NumPy equivalent
- [ ] Previous HyperFlowX version
- [ ] Other libraries (specify which)
- [ ] Theoretical optimum

### System Load
- **Concurrent Processes**: [Other running applications]
- **Thermal Throttling**: [CPU/GPU temperature issues]
- **Power Management**: [Performance vs. balanced mode]

### Proposed Solution
If you have ideas for optimization:
- **Algorithm Changes**: [Different approach?]
- **Implementation**: [Specific optimizations?]
- **Configuration**: [Parameter tuning?]

### Additional Context
- Screenshots of profiler output
- Links to related performance discussions
- Academic papers or benchmarks
- Hardware-specific considerations

### Checklist
- [ ] I have run the built-in benchmark (`python hyperflowx/benchmark.py`)
- [ ] I have provided complete system specifications
- [ ] I have included reproducible test code
- [ ] I have compared against reference implementations
- [ ] I have verified this isn't a known limitation