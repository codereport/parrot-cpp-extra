# AresDB Expand Example

This example demonstrates the AresDB expand functionality working locally with lightweight data structures, without requiring the full AresDB repository.

## Overview

The `expand` function implements a key operation from AresDB that expands dimension vectors based on count arrays and index vectors. It performs the following operations:

1. **Count Reduction**: Sums up counts from a base count array using index indirection
2. **Exclusive Scan**: Computes output offsets for each input element  
3. **Scatter**: Places elements into their corresponding output positions based on counts
4. **Max Scan**: Fills in gaps using maximum scan operation
5. **Dimension Copy**: Copies dimension values from input to output using custom iterators

## Key Components

### Data Structures

- **`DimensionVector`**: Represents dimension data with pointers to value arrays
- **`IndexCountIterator`**: Custom iterator that reads counts using index indirection
- **`DimensionColumnPermutateIterator`**: Iterator for reading dimension columns with permutation
- **`DimensionColumnOutputIterator`**: Iterator for writing to output dimension columns

### Execution Policy

The code uses a flexible execution policy system that supports different runtime environments:

```cpp
#ifdef RUN_ON_DEVICE
#  ifdef USE_RMM
#    define GET_EXECUTION_POLICY(cudaStream) \
       rmm::exec_policy(cudaStream)->on(cudaStream)
#  else
#    define GET_EXECUTION_POLICY(cudaStream) \
        thrust::cuda::par.on(cudaStream)
#  endif
#else
#  define GET_EXECUTION_POLICY(cudaStream) thrust::host
#endif
```

This allows the same code to run on:
- **CPU with host execution** (current implementation)
- **GPU with CUDA** (when compiled with `-DRUN_ON_DEVICE`)
- **GPU with RMM memory management** (when compiled with `-DRUN_ON_DEVICE -DUSE_RMM`)

### Algorithm Implementation

The expand function maintains the original algorithmic structure from AresDB while providing CPU fallbacks for Thrust algorithms:

- `thrust::reduce` → Sum reduction
- `thrust::exclusive_scan` → Prefix sum for offsets  
- `thrust::scatter_if` → Conditional scatter operation
- `thrust::inclusive_scan` → Maximum scan for gap filling
- `thrust::copy` → Dimension value copying

## Building and Running

### Using CMake (Current)
```bash
# From the project root
cd build
cmake --build . --target aresdb_expand_example -j$(nproc)
./aresdb_expand_example
```

### Alternative: Parrot.hpp Version
```bash
# From the project root  
cd build
cmake --build . --target aresdb_expand_parrot -j$(nproc)
./aresdb_expand_parrot
```

## Test Results

The example includes comprehensive tests that verify:

✓ **IndexCountIterator**: Index-based count access  
✓ **Basic Expand**: Core expansion functionality  
✓ **Pre-occupied Output**: Handling existing output data  
✓ **Capacity Limits**: Proper bounds checking  

All tests pass successfully, demonstrating that the lightweight implementation correctly replicates the AresDB expand behavior.

## Files

- **`expand.cu`**: Main expand function implementation with integrated tests
- **`expand_parrot.cu`**: Alternative implementation using parrot.hpp high-level array library
- **`aresdb_types.hpp`**: Lightweight data structures and iterator implementations
- **`README.md`**: This documentation file

The example demonstrates both a low-level Thrust implementation (expand.cu) and a high-level array programming approach (expand_parrot.cu).
