import cupy as cp


def max_abs_delta(a):
    return cp.max(cp.abs(cp.diff(a)))


def main():
    N = 10000
    rng = cp.random.default_rng(42)
    a = rng.uniform(0, 1000, N, dtype=cp.float32)
    result = max_abs_delta(a)
    print(f"max abs delta: {result}")
    return 0


if __name__ == "__main__":
    main()
