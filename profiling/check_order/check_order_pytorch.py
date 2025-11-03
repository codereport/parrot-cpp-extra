import torch


def check_order(ints):
    return torch.where(torch.sort(ints)[0] != ints)[0]


def main():
    N = 100000
    ints = torch.randint(0, N, (N,), device="cuda")
    result = check_order(ints)
    print(f"Out of order indices: {result}")
    print(f"Total: {len(result)} elements out of order")
    return 0


if __name__ == "__main__":
    main()
