import jax
import jax.numpy as jnp


def naive_softmax(x):
    x_max = jnp.max(x, axis=1, keepdims=True)
    z = x - x_max
    numerator = jnp.exp(z)
    denominator = jnp.sum(numerator, axis=1, keepdims=True)
    ret = numerator / denominator
    return ret


input_tensor = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.float32)

result = naive_softmax(input_tensor)
print(f"Result: {result}")
print(f"Result device: {result.device}")
print(f"JAX devices: {jax.devices()}")
