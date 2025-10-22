import tensorflow as tf


@tf.function(jit_compile=True)
def fused_softmax_jit(x):
    x_max = tf.reduce_max(x, axis=1, keepdims=True)
    z = x - x_max
    numerator = tf.exp(z)
    denominator = tf.reduce_sum(numerator, axis=1, keepdims=True)
    ret = numerator / denominator
    return ret


# Ensure TensorFlow uses GPU
with tf.device('/GPU:0'):
    input_tensor = tf.reshape(tf.range(10000, dtype=tf.float32), [100, 100])

    # First call will trigger JIT compilation
    result = fused_softmax_jit(input_tensor)
    print(f"Result: {result}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
