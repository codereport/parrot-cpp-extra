import torch


def naive_running_avg(data):
    # Compute cumulative sum
    cumsum = torch.cumsum(data, dim=0)
    # Create range [1, 2, 3, ..., n] for division
    divisor = torch.arange(1,
                           data.shape[0] + 1,
                           dtype=torch.float32,
                           device=data.device)
    # Compute running average
    running_avg = cumsum / divisor
    return running_avg


# Create data equivalent to parrot::range(10000) - [0, 1, 2, ..., 9999]
data = torch.arange(10000, dtype=torch.float32).to("cuda")

result = naive_running_avg(data)
print(f"Running average shape: {result.shape}")
print(f"First 10 values: {result[:10]}")
print(f"Last 10 values: {result[-10:]}")
print(f"Device: {result.device}")
