import tensorflow as tf


def naive_softmax(x):
    x_max = tf.reduce_max(x, axis=1, keepdims=True)
    z = x - x_max
    numerator = tf.exp(z)
    denominator = tf.reduce_sum(numerator, axis=1, keepdims=True)
    ret = numerator / denominator
    return ret


# Ensure TensorFlow uses GPU
with tf.device('/GPU:0'):
    input_tensor = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)

    result = naive_softmax(input_tensor)
    print(f"Result: {result}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
