import torch


def naive_softmax(x):
    x_max = x.max(dim=1)[0]
    z = x - x_max[:, None]
    numerator = torch.exp(z)
    denominator = numerator.sum(dim=1)
    ret = numerator / denominator[:, None]
    return ret


input_tensor = torch.arange(10000, dtype=torch.float32).reshape(100,
                                                                100).to("cuda")

result = naive_softmax(input_tensor)
print(f"Result: {result}")
