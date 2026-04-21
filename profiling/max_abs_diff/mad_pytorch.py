import torch


def max_abs_delta(a):
    return torch.diff(a).abs().max()


def main():
    N = 10000
    a = torch.rand(N, device="cuda") * 1000
    result = max_abs_delta(a)
    print(f"max abs delta: {result}")
    return 0


if __name__ == "__main__":
    main()
