import cupy as cp
import numpy as np
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "mad_data.bin"
N = 10000


def max_abs_delta(a):
    return cp.max(cp.abs(cp.diff(a)))


def main():
    a = cp.asarray(np.fromfile(DATA_FILE, dtype=np.int32))
    result = max_abs_delta(a)
    print(f"max abs delta: {result}")
    return 0


if __name__ == "__main__":
    main()
