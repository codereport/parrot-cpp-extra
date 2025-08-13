#!/usr/bin/env python3
import tensorflow as tf


@tf.function
def compute_outer_sum(arr):
    # Create outer sum using broadcasting
    return tf.reduce_sum(tf.add(arr[:, None], arr[None, :]), axis=1)


def main():
    arr = tf.range(1000, dtype=tf.float32)
    result = compute_outer_sum(arr)
    print(result)


if __name__ == "__main__":
    main()
