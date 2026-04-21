import jax
import jax.numpy as jnp


def max_abs_delta(a):
    return jnp.max(jnp.abs(jnp.diff(a)))


def main():
    N = 10000
    key = jax.random.PRNGKey(42)
    a = jax.random.uniform(key, (N,), minval=0, maxval=1000)
    result = max_abs_delta(a)
    print(f"max abs delta: {result}")
    return 0


if __name__ == "__main__":
    main()
