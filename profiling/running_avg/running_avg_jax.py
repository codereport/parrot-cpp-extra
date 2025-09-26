import jax
import jax.numpy as jnp


def naive_running_avg(data):
    # Compute cumulative sum
    cumsum = jnp.cumsum(data)
    # Create range [1, 2, 3, ..., n] for division
    divisor = jnp.arange(1, data.shape[0] + 1, dtype=jnp.float32)
    # Compute running average
    running_avg = cumsum / divisor
    return running_avg


# Create data equivalent to parrot::range(10000) - [0, 1, 2, ..., 9999]
data = jnp.arange(10000, dtype=jnp.float32)

result = naive_running_avg(data)
print(f"Running average shape: {result.shape}")
print(f"First 10 values: {result[:10]}")
print(f"Last 10 values: {result[-10:]}")
print(f"Result device: {result.device}")
print(f"JAX devices: {jax.devices()}")
