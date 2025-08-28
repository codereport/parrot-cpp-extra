#ifndef ARESDB_TYPES_HPP
#define ARESDB_TYPES_HPP

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <cstdint>

// Constants
#define NUM_DIM_WIDTH 4

// GPU-only execution policy - always use CUDA
#define GET_EXECUTION_POLICY(cudaStream) thrust::cuda::par.on(cudaStream)

namespace ares {
// Always use device vectors for GPU execution
template <typename T>
using device_vector = thrust::device_vector<T>;

template <typename T>
using host_vector = thrust::host_vector<T>;
}  // namespace ares

// DimensionVector structure representing dimension data
struct DimensionVector {
    void** DimValues;  // Array of pointers to dimension values for each width
    uint32_t VectorCapacity;                     // Total capacity of the vector
    uint32_t NumDimsPerDimWidth[NUM_DIM_WIDTH];  // Number of dimensions per
                                                 // width category

    DimensionVector() : DimValues(nullptr), VectorCapacity(0) {
        for (int i = 0; i < NUM_DIM_WIDTH; i++) { NumDimsPerDimWidth[i] = 0; }
    }

    DimensionVector(void** dimValues, uint32_t capacity, uint32_t* numDims)
      : DimValues(dimValues), VectorCapacity(capacity) {
        for (int i = 0; i < NUM_DIM_WIDTH; i++) {
            NumDimsPerDimWidth[i] = numDims[i];
        }
    }
};

// Iterator that reads counts from baseCounts array using indices from
// indexVector
class IndexCountIterator {
   private:
    uint32_t* baseCounts_;
    uint32_t* indexVector_;

   public:
    typedef uint32_t value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const uint32_t* pointer;
    typedef uint32_t reference;  // Return by value, not reference
    typedef std::random_access_iterator_tag iterator_category;

    __host__ __device__ IndexCountIterator(uint32_t* baseCounts,
                                           uint32_t* indexVector)
      : baseCounts_(baseCounts), indexVector_(indexVector) {}

    __host__ __device__ uint32_t operator[](difference_type n) const {
        return baseCounts_[indexVector_[n]];
    }

    __host__ __device__ reference operator*() const {
        return baseCounts_[indexVector_[0]];
    }

    __host__ __device__ IndexCountIterator operator+(difference_type n) const {
        IndexCountIterator result = *this;
        result.indexVector_ += n;
        return result;
    }

    __host__ __device__ IndexCountIterator& operator+=(difference_type n) {
        indexVector_ += n;
        return *this;
    }

    __host__ __device__ IndexCountIterator& operator++() {
        ++indexVector_;
        return *this;
    }

    __host__ __device__ IndexCountIterator operator-(difference_type n) const {
        return *this + (-n);
    }

    __host__ __device__ difference_type
    operator-(const IndexCountIterator& other) const {
        return indexVector_ - other.indexVector_;
    }

    __host__ __device__ bool operator==(const IndexCountIterator& other) const {
        return indexVector_ == other.indexVector_;
    }

    __host__ __device__ bool operator!=(const IndexCountIterator& other) const {
        return !(*this == other);
    }
};

// Iterator for reading dimension columns with permutation
class DimensionColumnPermutateIterator {
   private:
    void** dimValues_;
    uint32_t* indexVector_;
    uint32_t inputCapacity_;
    uint32_t outputLen_;
    uint32_t* numDimsPerWidth_;
    mutable uint32_t currentPos_;

   public:
    typedef uint8_t value_type;
    typedef std::ptrdiff_t difference_type;
    typedef uint8_t* pointer;
    typedef uint8_t& reference;
    typedef std::random_access_iterator_tag iterator_category;

    __host__ __device__
    DimensionColumnPermutateIterator(void** dimValues,
                                     uint32_t* indexVector,
                                     uint32_t inputCapacity,
                                     uint32_t outputLen,
                                     uint32_t* numDimsPerWidth)
      : dimValues_(dimValues),
        indexVector_(indexVector),
        inputCapacity_(inputCapacity),
        outputLen_(outputLen),
        numDimsPerWidth_(numDimsPerWidth),
        currentPos_(0) {}

    __host__ __device__ uint8_t operator[](difference_type n) const {
        // Calculate total number of dimensions
        uint32_t totalDims = 0;
        for (int w = 0; w < NUM_DIM_WIDTH; w++) {
            totalDims += numDimsPerWidth_[w];
        }

        // Simple linear indexing: n goes from 0 to (totalDims * 2 * outputLen -
        // 1) Layout: [dim0_elem0_byte0, dim0_elem0_byte1, dim1_elem0_byte0,
        // dim1_elem0_byte1, dim0_elem1_byte0, ...]
        uint32_t groupIndex  = n / (totalDims *
                                   2);  // Which element group (0, 1, 2, ...)
        uint32_t withinGroup = n %
                               (totalDims * 2);  // Position within the group
        uint32_t dimIndex = withinGroup / 2;  // Which dimension (0, 1, 2, ...)
        uint32_t byteOffset = withinGroup % 2;  // Which byte (0=low, 1=high)
        uint32_t elemIndex  = groupIndex;       // Which output element

        // Get the actual input index from indexVector
        uint32_t actualInputIndex = indexVector_[elemIndex];

        // Access the dimension value
        uint8_t* dimData = static_cast<uint8_t*>(dimValues_[dimIndex]);
        return dimData[actualInputIndex * 2 + byteOffset];
    }

    __host__ __device__ uint8_t operator*() const {
        return (*this)[currentPos_];
    }

    __host__ __device__ DimensionColumnPermutateIterator
    operator+(difference_type n) const {
        DimensionColumnPermutateIterator result = *this;
        result.currentPos_ += n;
        return result;
    }

    __host__ __device__ DimensionColumnPermutateIterator& operator+=(
      difference_type n) {
        currentPos_ += n;
        return *this;
    }

    __host__ __device__ DimensionColumnPermutateIterator& operator++() {
        ++currentPos_;
        return *this;
    }

    __host__ __device__ difference_type
    operator-(const DimensionColumnPermutateIterator& other) const {
        return currentPos_ - other.currentPos_;
    }

    __host__ __device__ bool operator==(
      const DimensionColumnPermutateIterator& other) const {
        return currentPos_ == other.currentPos_;
    }

    __host__ __device__ bool operator!=(
      const DimensionColumnPermutateIterator& other) const {
        return !(*this == other);
    }
};

// Iterator for writing to output dimension columns
class DimensionColumnOutputIterator {
   private:
    void** dimValues_;
    uint32_t outputCapacity_;
    uint32_t outputLen_;
    uint32_t* numDimsPerWidth_;
    uint32_t outputOffset_;
    mutable uint32_t currentPos_;

   public:
    typedef uint8_t value_type;
    typedef std::ptrdiff_t difference_type;
    typedef uint8_t* pointer;
    typedef uint8_t& reference;
    typedef std::random_access_iterator_tag iterator_category;

    __host__ __device__ DimensionColumnOutputIterator(void** dimValues,
                                                      uint32_t outputCapacity,
                                                      uint32_t outputLen,
                                                      uint32_t* numDimsPerWidth,
                                                      uint32_t outputOffset)
      : dimValues_(dimValues),
        outputCapacity_(outputCapacity),
        outputLen_(outputLen),
        numDimsPerWidth_(numDimsPerWidth),
        outputOffset_(outputOffset),
        currentPos_(0) {}

    __host__ __device__ DimensionColumnOutputIterator& operator[](
      difference_type n) const {
        return const_cast<DimensionColumnOutputIterator&>(*this);
    }

    __host__ __device__ DimensionColumnOutputIterator& operator*() const {
        return const_cast<DimensionColumnOutputIterator&>(*this);
    }

    __host__ __device__ DimensionColumnOutputIterator& operator=(
      uint8_t value) {
        // Calculate total number of dimensions
        uint32_t totalDims = 0;
        for (int w = 0; w < NUM_DIM_WIDTH; w++) {
            totalDims += numDimsPerWidth_[w];
        }

        // Simple linear indexing: currentPos_ goes from 0 to (totalDims * 2 *
        // outputLen - 1) Layout: [dim0_elem0_byte0, dim0_elem0_byte1,
        // dim1_elem0_byte0, dim1_elem0_byte1, dim0_elem1_byte0, ...]
        uint32_t groupIndex = currentPos_ /
                              (totalDims *
                               2);  // Which element group (0, 1, 2, ...)
        uint32_t withinGroup = currentPos_ %
                               (totalDims * 2);  // Position within the group
        uint32_t dimIndex = withinGroup / 2;  // Which dimension (0, 1, 2, ...)
        uint32_t byteOffset = withinGroup % 2;  // Which byte (0=low, 1=high)
        uint32_t elemIndex  = groupIndex;       // Which output element

        // Calculate the actual output position (with offset)
        uint32_t actualOutputIndex = outputOffset_ + elemIndex;

        // Optional debug output (disabled for production)
        // if (currentPos_ < 8) {
        //     printf("OUTPUT[%d]: Writing value %d to dim%d[%d][%d] (pos
        //     %d)\n",
        //            currentPos_, (int)value, dimIndex, actualOutputIndex,
        //            byteOffset, currentPos_);
        // }

        // Write to the dimension value
        uint8_t* dimData = static_cast<uint8_t*>(dimValues_[dimIndex]);
        dimData[actualOutputIndex * 2 + byteOffset] = value;

        return *this;
    }

    __host__ __device__ DimensionColumnOutputIterator
    operator+(difference_type n) const {
        DimensionColumnOutputIterator result = *this;
        result.currentPos_ += n;
        return result;
    }

    __host__ __device__ DimensionColumnOutputIterator& operator+=(
      difference_type n) {
        currentPos_ += n;
        return *this;
    }

    __host__ __device__ DimensionColumnOutputIterator& operator++() {
        ++currentPos_;
        return *this;
    }

    __host__ __device__ difference_type
    operator-(const DimensionColumnOutputIterator& other) const {
        return currentPos_ - other.currentPos_;
    }

    __host__ __device__ bool operator==(
      const DimensionColumnOutputIterator& other) const {
        return currentPos_ == other.currentPos_;
    }

    __host__ __device__ bool operator!=(
      const DimensionColumnOutputIterator& other) const {
        return !(*this == other);
    }
};

#endif  // ARESDB_TYPES_HPP
