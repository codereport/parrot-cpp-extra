#!/usr/bin/env python3
import cupy as cp


def main():
    arr = cp.arange(1000)
    result = cp.add.outer(arr, arr).sum(axis=1)
    print(result)


if __name__ == "__main__":
    main()
