import tensorflow as tf


@tf.function(jit_compile=True)
def fused_running_avg_jit(data):
    # Compute cumulative sum
    cumsum = tf.cumsum(data)
    # Create range [1, 2, 3, ..., n] for division
    divisor = tf.range(1, tf.shape(data)[0] + 1, dtype=tf.float32)
    # Compute running average
    running_avg = cumsum / divisor
    return running_avg


# Ensure TensorFlow uses GPU
with tf.device('/GPU:0'):
    # Create data equivalent to parrot::range(10000) - [0, 1, 2, ..., 9999]
    data = tf.range(10000, dtype=tf.float32)

    # First call will trigger JIT compilation
    result = fused_running_avg_jit(data)
    print(f"Running average shape: {result.shape}")
    print(f"First 10 values: {result[:10]}")
    print(f"Last 10 values: {result[-10:]}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
