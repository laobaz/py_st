import torch
from torch import nn


class Tudui(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = input + 1
        return output


tudui = Tudui()
x = torch.Tensor(1.0)  # 加载张量
output = tudui(x)
print(output)
