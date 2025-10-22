import torch
import torch.nn.functional as F


# @torch.compile(backend="inductor", mode="max-autotune")
def fused_softmax_v1(x):
    return F.softmax(x, dim=1)


small_input = torch.arange(10000, dtype=torch.float32,
                           device="cuda").reshape(100, 100)
small_result = fused_softmax_v1(small_input)
print(f"Small result: {small_result}")
