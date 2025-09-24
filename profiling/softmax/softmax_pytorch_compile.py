import torch


@torch.compile(backend="inductor", mode="max-autotune")
def fused_softmax_compiled(x):
    # Using logsumexp - fastest manual implementation for numerical stability
    return torch.exp(x - torch.logsumexp(x, dim=1, keepdim=True))


input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],
                            dtype=torch.float32).to("cuda")

# First call will trigger compilation
result = fused_softmax_compiled(input_tensor)

print(f"Result: {result}")
