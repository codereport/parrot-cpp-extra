import torch
import torch.jit


@torch.jit.script
def fused_softmax_jit(x):
    x_max = torch.max(x, dim=1, keepdim=True)[0]
    z = x - x_max
    numerator = torch.exp(z)
    denominator = torch.sum(numerator, dim=1, keepdim=True)
    ret = numerator / denominator
    return ret


input_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]],
                            dtype=torch.float32).to("cuda")

# First call will trigger JIT compilation
result = fused_softmax_jit(input_tensor)

print(f"Result: {result}")
