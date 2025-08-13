#!/usr/bin/env python3
import torch


@torch.jit.script
def outer_sum_jit(arr: torch.Tensor) -> torch.Tensor:
    """JIT compiled outer sum operation."""
    return (arr[:, None] + arr[None, :]).sum(dim=1)


def main():
    device = torch.device("cuda")
    arr = torch.arange(1000, device=device)
    result = outer_sum_jit(arr)
    print(result)


if __name__ == "__main__":
    main()
