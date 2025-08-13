#!/usr/bin/env python3
import jax.numpy as jnp


def main():
    arr = jnp.arange(1000)
    result = jnp.add.outer(arr, arr).sum(axis=1)
    print(result)


if __name__ == "__main__":
    main()
