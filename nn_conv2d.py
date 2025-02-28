import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Conv2d

from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

test_Set=torchvision.datasets.CIFAR10("./dataset",False,download=True,transform=torchvision.transforms.ToTensor())
test_data=DataLoader(test_Set,64)
class TuDui(nn.Module):
    def  __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=Conv2d(in_channels=3,out_channels=6,
                          kernel_size=3,stride=1,padding=0)

    def forward(self,x):
        x=self.conv1(x)
        return x


tudui=TuDui()
step=0
writer=SummaryWriter("conv2D")
for data in test_data:
    imgs,targets=data
    output=tudui(imgs)
    print(output.shape)
    output=torch.reshape(output,(-1,3,30,30))
    writer.add_images("conv2D",output,step)
    step=step+1
writer.close()
