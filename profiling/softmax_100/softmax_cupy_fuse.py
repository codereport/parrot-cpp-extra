import cupy as cp


@cp.fuse()
def fused_softmax(x):
    x_max = cp.max(x, axis=1, keepdims=True)
    z = x - x_max
    numerator = cp.exp(z)
    denominator = cp.sum(numerator, axis=1, keepdims=True)
    ret = numerator / denominator
    return ret


input_tensor = cp.arange(10000, dtype=cp.float32).reshape(100, 100)

# First call will generate and cache the fused kernel
result = fused_softmax(input_tensor)
print(f"Result: {result}")
print(f"CuPy device: {cp.cuda.Device()}")
