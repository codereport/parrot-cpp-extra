import jax
import jax.numpy as jnp


@jax.jit
def fused_softmax(x):
    x_max = jnp.max(x, axis=1, keepdims=True)
    z = x - x_max
    numerator = jnp.exp(z)
    denominator = jnp.sum(numerator, axis=1, keepdims=True)
    ret = numerator / denominator
    return ret


input_tensor = jnp.arange(10000, dtype=jnp.float32).reshape(100, 100)

result = fused_softmax(input_tensor)
print(f"Result: {result}")
