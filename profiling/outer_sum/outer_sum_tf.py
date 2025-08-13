#!/usr/bin/env python3
import tensorflow as tf


def main():
    arr = tf.range(1000, dtype=tf.float32)
    # Create outer sum using broadcasting
    result = tf.reduce_sum(tf.add(arr[:, None], arr[None, :]), axis=1)
    print(result)


if __name__ == "__main__":
    main()
