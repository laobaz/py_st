

import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d

from   torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

data_set=torchvision.datasets.CIFAR10("./dataset",False,transform=torchvision.transforms.ToTensor())
dataloader=DataLoader(data_set,64)

input=torch.tensor([[1,2,3,4,5],
                   [1,2,3,4,5],
                   [1,2,3,4,5,],
                    [1,2,3,4,5],
                    [1,2,3,4,5]],dtype=torch.float32)

input=torch.reshape(input,(-1,1,5,5))

class Tudui(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooling= MaxPool2d(kernel_size=3,ceil_mode=False)

    def forward(self,x):
        x=self.pooling(x)
        return x

tudui =Tudui()

step=0
writer=SummaryWriter("pooling")
for data in dataloader:
    imgs,targets=data
    output=tudui(imgs)
    writer.add_images("pooling",output,step)
    step=step+1
writer.close()