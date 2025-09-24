import cupy as cp


def naive_softmax(x):
    x_max = cp.max(x, axis=1, keepdims=True)
    z = x - x_max
    numerator = cp.exp(z)
    denominator = cp.sum(numerator, axis=1, keepdims=True)
    ret = numerator / denominator
    return ret


input_tensor = cp.array([[1, 2, 3], [4, 5, 6]], dtype=cp.float32)

result = naive_softmax(input_tensor)
print(f"Result: {result}")
print(f"CuPy device: {cp.cuda.Device()}")
