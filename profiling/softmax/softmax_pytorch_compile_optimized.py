import torch
import torch.nn.functional as F


# @torch.compile(backend="inductor", mode="max-autotune")
def fused_softmax_v1(x):
    return F.softmax(x, dim=1)


small_input = torch.tensor([[1, 2, 3], [4, 5, 6]],
                           dtype=torch.float32,
                           device="cuda")
small_result = fused_softmax_v1(small_input)
print(f"Small result: {small_result}")
