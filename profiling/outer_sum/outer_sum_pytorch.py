#!/usr/bin/env python3
import torch


def main():
    device = torch.device("cuda")
    arr = torch.arange(1000, device=device)
    result = (arr[:, None] + arr[None, :]).sum(dim=1)
    print(result)


if __name__ == "__main__":
    main()
