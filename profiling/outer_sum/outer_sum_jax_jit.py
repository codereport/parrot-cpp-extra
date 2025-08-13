#!/usr/bin/env python3
import jax.numpy as jnp
import jax


@jax.jit
def compute_outer_sum(arr):
    """JIT-compiled function to compute outer sum"""
    result = jnp.add.outer(arr, arr).sum(axis=1)
    return result


def main():
    arr = jnp.arange(1000)
    result = compute_outer_sum(arr)
    print(result)


if __name__ == "__main__":
    main()
