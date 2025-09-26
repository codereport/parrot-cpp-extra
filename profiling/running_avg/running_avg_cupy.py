import cupy as cp


def naive_running_avg(data):
    # Compute cumulative sum
    cumsum = cp.cumsum(data)
    # Create range [1, 2, 3, ..., n] for division
    divisor = cp.arange(1, data.shape[0] + 1, dtype=cp.float32)
    # Compute running average
    running_avg = cumsum / divisor
    return running_avg


# Create data equivalent to parrot::range(10000) - [0, 1, 2, ..., 9999]
data = cp.arange(10000, dtype=cp.float32)

result = naive_running_avg(data)
print(f"Running average shape: {result.shape}")
print(f"First 10 values: {result[:10]}")
print(f"Last 10 values: {result[-10:]}")
print(f"CuPy device: {cp.cuda.Device()}")
