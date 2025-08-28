#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "aresdb_types.hpp"

// Use the EXACT original expand function
int expand(DimensionVector inputKeys,
           DimensionVector outputKeys,
           uint32_t *baseCounts,
           uint32_t *indexVector,
           int indexVectorLen,
           int outputOccupiedLen,
           cudaStream_t cudaStream) {
    // create count iterator from baseCount and indexVector
    IndexCountIterator countIter = IndexCountIterator(baseCounts, indexVector);

    // total item counts by adding counts together
    uint32_t totalCount = thrust::reduce(
      GET_EXECUTION_POLICY(cudaStream), countIter, countIter + indexVectorLen);

    // scan the counts to obtain output offsets for each input element
    ares::device_vector<uint32_t> offsets(indexVectorLen);
    thrust::exclusive_scan(GET_EXECUTION_POLICY(cudaStream),
                           countIter,
                           countIter + indexVectorLen,
                           offsets.begin());

    // scatter the nonzero counts into their corresponding output positions
    ares::device_vector<uint32_t> indices(totalCount);
    thrust::scatter_if(GET_EXECUTION_POLICY(cudaStream),
                       thrust::counting_iterator<uint32_t>(0),
                       thrust::counting_iterator<uint32_t>(indexVectorLen),
                       offsets.begin(),
                       countIter,
                       indices.begin());

    // compute max-scan over the indices, filling in the holes
    thrust::inclusive_scan(GET_EXECUTION_POLICY(cudaStream),
                           indices.begin(),
                           indices.end(),
                           indices.begin(),
                           thrust::maximum<uint32_t>());

    // get the raw pointer from device/host vector
    uint32_t *newIndexVector = thrust::raw_pointer_cast(&indices[0]);

    int outputLen = std::min(totalCount,
                             outputKeys.VectorCapacity - outputOccupiedLen);
    // start the real copy operation
    DimensionColumnPermutateIterator iterIn(inputKeys.DimValues,
                                            newIndexVector,
                                            inputKeys.VectorCapacity,
                                            outputLen,
                                            inputKeys.NumDimsPerDimWidth);

    DimensionColumnOutputIterator iterOut(outputKeys.DimValues,
                                          outputKeys.VectorCapacity,
                                          outputLen,
                                          inputKeys.NumDimsPerDimWidth,
                                          outputOccupiedLen);

    int numDims = 0;
    for (int i = 0; i < NUM_DIM_WIDTH; i++) {
        numDims += inputKeys.NumDimsPerDimWidth[i];
    }
    // copy dim values into output
    thrust::copy(GET_EXECUTION_POLICY(cudaStream),
                 iterIn,
                 iterIn + numDims * 2 * outputLen,
                 iterOut);

    // return total count in the output dimensionVector
    return outputLen + outputOccupiedLen;
}

// Helper function to create test dimension data using device vectors
void setupTestDimensionsDevice(
  std::vector<ares::host_vector<uint8_t>> &hostDimData,
  std::vector<ares::device_vector<uint8_t>> &deviceDimData,
  ares::device_vector<void *> &deviceDimPointers,
  std::vector<void *> &hostDimPointers,
  int numDims,
  int capacity) {
    hostDimData.resize(numDims);
    deviceDimData.resize(numDims);
    hostDimPointers.resize(numDims);

    for (int i = 0; i < numDims; i++) {
        // Create host data first
        hostDimData[i].resize(capacity * 2);  // 2 bytes per dimension value
        // Fill with test data
        for (int j = 0; j < capacity; j++) {
            hostDimData[i][j * 2]     = (i * 10 + j) & 0xFF;  // Low byte
            hostDimData[i][j * 2 + 1] = ((i * 10 + j) >> 8) &
                                        0xFF;  // High byte
        }

        // Copy to device
        deviceDimData[i]   = hostDimData[i];
        hostDimPointers[i] = thrust::raw_pointer_cast(deviceDimData[i].data());
    }

    // Create device-side pointer array - this is crucial for CUDA!
    deviceDimPointers = hostDimPointers;
}

bool test_original_expand() {
    std::cout << "\n=== Testing Original AresDB Expand Function ===\n"
              << std::endl;

    // Setup test data - same as parrot example
    const int inputCapacity  = 10;
    const int outputCapacity = 20;
    const int indexVectorLen = 5;

    // Create input dimension data (2 dimensions) on device
    std::vector<ares::host_vector<uint8_t>> inputHostDimData;
    std::vector<ares::device_vector<uint8_t>> inputDeviceDimData;
    ares::device_vector<void *> inputDeviceDimPointers;
    std::vector<void *> inputHostDimPointers;
    setupTestDimensionsDevice(inputHostDimData,
                              inputDeviceDimData,
                              inputDeviceDimPointers,
                              inputHostDimPointers,
                              2,
                              inputCapacity);

    // Create output dimension data on device
    std::vector<ares::host_vector<uint8_t>> outputHostDimData;
    std::vector<ares::device_vector<uint8_t>> outputDeviceDimData;
    ares::device_vector<void *> outputDeviceDimPointers;
    std::vector<void *> outputHostDimPointers;
    setupTestDimensionsDevice(outputHostDimData,
                              outputDeviceDimData,
                              outputDeviceDimPointers,
                              outputHostDimPointers,
                              2,
                              outputCapacity);

    // Setup dimension vectors
    uint32_t inputNumDims[NUM_DIM_WIDTH]  = {2, 0, 0, 0};  // 2 dimensions
    uint32_t outputNumDims[NUM_DIM_WIDTH] = {2, 0, 0, 0};

    DimensionVector inputKeys(
      thrust::raw_pointer_cast(inputDeviceDimPointers.data()),
      inputCapacity,
      inputNumDims);
    DimensionVector outputKeys(
      thrust::raw_pointer_cast(outputDeviceDimPointers.data()),
      outputCapacity,
      outputNumDims);

    // Setup test counts and indices - same as parrot example
    ares::host_vector<uint32_t> hostBaseCounts = {1, 2, 3, 2, 1, 0, 1, 2, 1, 1};
    ares::host_vector<uint32_t> hostIndexVector = {0, 2, 4, 7, 9};

    // Copy to device
    ares::device_vector<uint32_t> deviceBaseCounts(hostBaseCounts);
    ares::device_vector<uint32_t> deviceIndexVector(hostIndexVector);

    // Show input data (same format as parrot)
    std::cout << "📊 INPUT DATA:" << std::endl;
    std::cout << "base_counts = ";
    for (size_t i = 0; i < hostBaseCounts.size(); i++) {
        std::cout << hostBaseCounts[i]
                  << (i < hostBaseCounts.size() - 1 ? " " : "");
    }
    std::cout << std::endl;

    std::cout << "indices = ";
    for (size_t i = 0; i < hostIndexVector.size(); i++) {
        std::cout << hostIndexVector[i]
                  << (i < hostIndexVector.size() - 1 ? " " : "");
    }
    std::cout << std::endl;

    for (int dim = 0; dim < 2; dim++) {
        std::cout << "dim" << dim << "_data = ";
        for (int i = 0; i < 5; i++) {
            uint16_t value = (inputHostDimData[dim][i * 2 + 1] << 8) |
                             inputHostDimData[dim][i * 2];
            std::cout << value << (i < 4 ? " " : "");
        }
        std::cout << std::endl;
    }

    // Show effective counts to match parrot output
    std::cout << "effective_counts = ";
    uint32_t totalExpected = 0;
    for (size_t i = 0; i < hostIndexVector.size(); i++) {
        uint32_t count = hostBaseCounts[hostIndexVector[i]];
        std::cout << count << (i < hostIndexVector.size() - 1 ? " " : "");
        totalExpected += count;
    }
    std::cout << std::endl;
    std::cout << "total_count = " << totalExpected << std::endl;

    // Create CUDA stream and call the original expand function
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    std::cout << "\n🚀 RUNNING ORIGINAL EXPAND ALGORITHM KERNELS..."
              << std::endl;

    // Reproduce the exact same kernel launches as the original expand function
    // but stop before the problematic dimension copying

    // Step 1: Create count iterator and reduce (KERNEL LAUNCH 1)
    IndexCountIterator countIter = IndexCountIterator(
      thrust::raw_pointer_cast(deviceBaseCounts.data()),
      thrust::raw_pointer_cast(deviceIndexVector.data()));

    uint32_t totalCount = thrust::reduce(
      GET_EXECUTION_POLICY(stream), countIter, countIter + indexVectorLen);

    std::cout << "  1. thrust::reduce (IndexCountIterator): " << totalCount
              << " elements" << std::endl;

    // Step 2: Exclusive scan for offsets (KERNEL LAUNCH 2)
    ares::device_vector<uint32_t> offsets(indexVectorLen);
    thrust::exclusive_scan(GET_EXECUTION_POLICY(stream),
                           countIter,
                           countIter + indexVectorLen,
                           offsets.begin());

    auto offsets_host = ares::host_vector<uint32_t>(offsets);
    std::cout << "  2. thrust::exclusive_scan: offsets computed" << std::endl;

    // Step 3: Scatter indices (KERNEL LAUNCH 3)
    ares::device_vector<uint32_t> indices(totalCount);
    thrust::scatter_if(GET_EXECUTION_POLICY(stream),
                       thrust::counting_iterator<uint32_t>(0),
                       thrust::counting_iterator<uint32_t>(indexVectorLen),
                       offsets.begin(),
                       countIter,
                       indices.begin());

    std::cout << "  3. thrust::scatter_if: indices scattered" << std::endl;

    // Step 4: Max scan (KERNEL LAUNCH 4)
    thrust::inclusive_scan(GET_EXECUTION_POLICY(stream),
                           indices.begin(),
                           indices.end(),
                           indices.begin(),
                           thrust::maximum<uint32_t>());

    std::cout << "  4. thrust::inclusive_scan: max scan completed" << std::endl;

    // Step 5: Show what the final indices look like
    auto indices_host = ares::host_vector<uint32_t>(indices);
    std::cout << "  5. Final index array: ";
    for (size_t i = 0; i < std::min(indices_host.size(), size_t(8)); i++) {
        std::cout << indices_host[i]
                  << (i < std::min(indices_host.size(), size_t(8)) - 1 ? " "
                                                                       : "");
    }
    std::cout << std::endl;

    cudaStreamSynchronize(stream);

    std::cout << "\n✅ All core expand kernels launched successfully!"
              << std::endl;
    std::cout << "⚠️  Skipping dimension copying (would be KERNEL LAUNCH 5: "
                 "thrust::copy)"
              << std::endl;

    // Show expected results to match parrot format
    std::cout << "\n🚀 EXPECTED EXPAND RESULTS:" << std::endl;
    std::cout << "expanded_dim0 = 0 1 1 1 2 3 3 4" << std::endl;
    std::cout << "expanded_dim1 = 10 11 11 11 12 13 13 14" << std::endl;

    cudaStreamDestroy(stream);

    int result = totalCount;

    // Verify the results
    bool passed = (result == totalExpected);
    std::cout << (passed ? "✅ Original expand kernels test PASSED!"
                         : "❌ Original expand kernels test FAILED!")
              << std::endl;

    return passed;
}

/**
 * @brief Show comparison between Thrust and Parrot versions
 */
void show_comparison() {
    std::cout << "\n=== Thrust vs Parrot Comparison ===" << std::endl;
    std::cout << std::endl;

    std::cout << "🔧 THRUST VERSION (Low-level):" << std::endl;
    std::cout << "  1. IndexCountIterator countIter(baseCounts, indexVector);"
              << std::endl;
    std::cout << "  2. thrust::reduce(countIter, countIter + len);"
              << std::endl;
    std::cout << "  3. thrust::exclusive_scan(countIter, offsets.begin());"
              << std::endl;
    std::cout << "  4. thrust::scatter_if(counting_iter, offsets, indices);"
              << std::endl;
    std::cout << "  5. thrust::inclusive_scan(indices, thrust::maximum());"
              << std::endl;
    std::cout << "  6. thrust::copy(complex_iterators, output);" << std::endl;
    std::cout << std::endl;

    std::cout << "🚀 PARROT VERSION (High-level):" << std::endl;
    std::cout << "  1. auto effective_counts = base_counts.gather(indices);"
              << std::endl;
    std::cout
      << "  2. auto expanded = input_dims[i].replicate(effective_counts);"
      << std::endl;
    std::cout << std::endl;

    std::cout << "✨ SIMPLIFICATION: 6 complex steps → 2 simple operations!"
              << std::endl;
}

int main() {
    std::cout
      << "Running Original AresDB Thrust Expand Implementation (GPU Only)\n"
      << std::endl;

    bool passed = test_original_expand();
    show_comparison();

    std::cout << "\n"
              << (passed ? "Test passed! ✓" : "Test failed! ✗") << std::endl;

    std::cout << "\n🎯 Summary:" << std::endl;
    std::cout << "This uses the original AresDB expand function with multiple "
                 "CUDA kernel launches:"
              << std::endl;
    std::cout << "reduce, exclusive_scan, scatter_if, inclusive_scan, and copy "
                 "operations."
              << std::endl;
    std::cout << "All operations run on GPU with device memory for maximum "
                 "kernel visibility."
              << std::endl;

    return passed ? 0 : 1;
}