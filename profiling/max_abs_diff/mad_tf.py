import tensorflow as tf


def max_abs_delta(a):
    with tf.device("/GPU:0"):
        return tf.reduce_max(tf.abs(a[1:] - a[:-1]))


def main():
    N = 10000
    with tf.device("/GPU:0"):
        a = tf.random.uniform((N,), minval=0, maxval=1000)
    result = max_abs_delta(a)
    print(f"max abs delta: {result}")
    return 0


if __name__ == "__main__":
    main()
