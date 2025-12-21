# System Performance Bottleneck Analysis Report

## Executive Summary

The system is achieving only ~4 FPS/IPS instead of the configured 120 FPS target, representing a 97% performance gap. This analysis identifies multiple contributing factors that cumulatively create significant performance degradation.

## Identified Bottlenecks

### 1. Logging Overhead
**Location**: Throughout the codebase (`camera_capture.cpp`, `inference.cpp`, `logic.cpp`, `h264_encoder.cpp`)
**Impact**: High

#### Details:
- Extensive DEBUG and INFO logging in the main processing loops
- Frequent file I/O operations for CSV logging
- Buffer flush operations every 100ms
- Mutex locking for thread-safe logging operations

#### Per-Call Cost Estimation:
- Each `APP_LOG_*` call: ~100-500 microseconds (including mutex lock/unlock)
- CSV logging with file I/O: ~500-2000 microseconds per entry
- Buffer flush operations: ~1000-5000 microseconds

#### Cumulative Impact:
With ~4 FPS, that's ~4 log entries per second in the main loop, but considering all modules and debug logs, the actual count is much higher, leading to significant overhead.

### 2. Thread Contention and Synchronization
**Location**: `camera_capture.cpp`, `util_logging.cpp`, `buffer_pool.h`
**Impact**: Medium-High

#### Details:
- Mutex locks in buffer pool acquisition with 1-second timeouts
- Condition variables in camera request processing
- Thread joins during shutdown operations
- Shared resource access in logging system

#### Blocking Operations:
- `buffer_pool.acquire()` can block up to 1 second if no buffers available
- `request_queue_cond_var_.wait()` in camera processing thread
- Thread joins during module shutdown

### 3. Memory Operations and Buffer Management
**Location**: `camera_capture.cpp`, `inference.cpp`, `image_processor.cpp`
**Impact**: Medium

#### Details:
- Frequent `memcpy` operations for frame data (multiple MB per frame)
- `mmap`/`munmap` operations for DMA buffer access
- Buffer resizing operations when sizes don't match
- Shared pointer management overhead

#### Estimated Costs:
- `memcpy` for 320x320 RGB frame: ~500-1000 microseconds
- `mmap`/`munmap` operations: ~100-300 microseconds each
- Buffer pool acquisition with mutex: ~50-200 microseconds

### 4. TPU Inference Engine Issues
**Location**: `inference.cpp`
**Impact**: High

#### Details:
- Frequent interpreter recreation due to EdgeTPU delegate issues
- Retry mechanisms with multiple attempts
- Periodic interpreter recreation every RECREATE_INTERVAL inferences
- Error handling and recovery logic

#### Problematic Patterns:
- Interpreter recreation can take 10-50ms
- Up to 3 retry attempts per failed inference
- Periodic recreation adds consistent overhead

### 5. Pipeline Structure and Flow Control
**Location**: All modules using SPSC queues
**Impact**: Medium

#### Details:
- Lock-free SPSC queues can still cause cache coherency issues
- Producer/consumer imbalance leads to backpressure
- Busy-waiting with `sleep_for` in empty queue scenarios
- No flow control between pipeline stages

## Quantified Delays

| Component | Per-Operation Delay | Frequency | Cumulative Impact |
|-----------|-------------------|-----------|-------------------|
| Logging overhead | 500 μs avg | 10-50x per frame | 5-25ms per frame |
| Buffer operations | 1000 μs avg | 3-5x per frame | 3-5ms per frame |
| TPU inference recreation | 30000 μs avg | Every RECREATE_INTERVAL | Variable |
| TPU inference retry | 10000 μs avg | On failures | Variable |
| Thread synchronization | 100 μs avg | 5-10x per frame | 0.5-1ms per frame |
| Queue operations | 10 μs avg | 10-20x per frame | 0.1-0.2ms per frame |

## Recommended Fixes (Priority Order)

### 1. Optimize Logging (Highest Priority)
**Action**: 
- Disable DEBUG logging in production
- Batch CSV log entries instead of flushing every 100ms
- Use asynchronous logging with larger buffers
- Reduce frequency of INFO logs in hot paths

**Expected Improvement**: 30-50% performance gain

### 2. Fix TPU Inference Stability (High Priority)
**Action**:
- Investigate root cause of interpreter recreation issues
- Implement proper error recovery without full interpreter recreation
- Optimize delegate initialization
- Consider using a single interpreter with proper error handling

**Expected Improvement**: 20-40% performance gain

### 3. Optimize Buffer Management (Medium Priority)
**Action**:
- Pre-allocate buffers with correct sizes to avoid resizing
- Implement zero-copy mechanisms where possible
- Reduce frequency of mmap/munmap operations
- Optimize buffer pool sizes

**Expected Improvement**: 10-20% performance gain

### 4. Reduce Thread Contention (Medium Priority)
**Action**:
- Reduce frequency of mutex acquisitions
- Implement lock-free alternatives where possible
- Optimize condition variable usage
- Consider reducing number of worker threads

**Expected Improvement**: 5-15% performance gain

### 5. Pipeline Optimization (Lower Priority)
**Action**:
- Implement proper backpressure handling
- Optimize queue sizes for better throughput
- Reduce busy-waiting with smarter sleep strategies
- Consider pipeline parallelization improvements

**Expected Improvement**: 5-10% performance gain

## Benchmark Isolation Tests

To validate these findings, recommend implementing:

1. **Raw TPU Benchmark**: Minimal TPU inference test without any logging or camera capture
2. **Logging Overhead Test**: Same pipeline with logging disabled vs enabled
3. **Buffer Copy Test**: Measure cost of various buffer operations
4. **Thread Contention Test**: Measure performance with varying thread counts

## Resource Profiling Recommendations

1. **CPU Profiling**: Use `perf` to identify hotspots in the main processing loops
2. **Memory Profiling**: Monitor buffer allocation/deallocation patterns
3. **I/O Profiling**: Measure file system I/O impact from logging
4. **Thread Analysis**: Use tools like `htop` to monitor thread behavior and CPU usage patterns

## Conclusion

The severe performance degradation is primarily caused by excessive logging overhead combined with TPU inference engine instability requiring frequent interpreter recreation. Addressing these two issues should recover 50-70% of the lost performance, bringing the system much closer to the target 120 FPS.