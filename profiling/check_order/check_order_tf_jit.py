import tensorflow as tf


@tf.function
def check_order(ints):
    return tf.where(tf.sort(ints) != ints)[:, 0]


def main():
    N = 100000
    with tf.device("/GPU:0"):
        ints = tf.random.uniform((N,), 0, N, dtype=tf.int32)
        result = check_order(ints)
        print(f"Out of order indices: {result}")
        print(f"Total: {len(result)} elements out of order")
    return 0


if __name__ == "__main__":
    main()
