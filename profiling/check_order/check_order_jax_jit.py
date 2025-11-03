import jax
import jax.numpy as jnp


@jax.jit
def check_order(ints):
    return jnp.where(jnp.sort(ints) != ints)[0]


def main():
    N = 100000
    key = jax.random.PRNGKey(0)
    ints = jax.random.randint(key, (N,), 0, N)
    result = check_order(ints)
    result.block_until_ready()
    print(f"Out of order indices: {result}")
    print(f"Total: {len(result)} elements out of order")
    return 0


if __name__ == "__main__":
    main()
