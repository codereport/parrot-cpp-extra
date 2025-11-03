import cupy as cp


@cp.fuse()
def check_order(ints):
    return cp.where(cp.sort(ints) != ints)[0]


def main():
    N = 100000
    ints = cp.random.randint(0, N, size=N)
    result = check_order(ints)
    print(f"Out of order indices: {result}")
    print(f"Total: {len(result)} elements out of order")
    return 0


if __name__ == "__main__":
    main()
