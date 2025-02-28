import torch
import torch.nn.functional as F
input=torch.tensor([[1,2,3,4,5],
                   [1,2,3,4,5],
                   [1,2,3,4,5,],
                    [1,2,3,4,5],
                    [1,2,3,4,5]])

kernel=torch.tensor([[1,2,3],
                     [1,2,3],
                     [1,2,3]])
print(input.shape)
# Expected 3D (unbatched) or 4D (batched) input to conv2d, but got input of size: [5, 5]
# 要求4D，只有2D
input=torch.reshape(input,(1,1,5,5))
kernel=torch.reshape(kernel,(1,1,3,3))
output=F.conv2d(input,kernel,stride=1,padding=1)
print(output)