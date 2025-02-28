import torch
import torchvision
from torch import nn

from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader

# data_set=torchvision.datasets.CIFAR10("./dataset",False,transform=torchvision.transforms.ToTensor())
# dataloader=DataLoader(data_set,64)
class Tudui(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=Conv2d(3,32,5,1,2)
        self.Maxpool1=MaxPool2d(2)
        self.conv2=Conv2d(32,32,5,1,2)
        self.Maxpool2=MaxPool2d(2)
        self.conv3=Conv2d(32,64,5,1,2)
        self.Maxpool3=MaxPool2d(2)
        self.flatten=Flatten()
        self.linear1=Linear(1024,64)
        self.linear2=Linear(64,10)

        self.model1=Sequential(
            Conv2d(3,32,5,1,2)
            # .....
        )
    def forward(self,x):
     return   (
        self.linear2(
        self.linear1(
        self.flatten(
        self.Maxpool3(
        self.conv3(
        self.Maxpool2(
        self.conv2(
        self.Maxpool1(
        self.conv1(x))))))))))



# loss=nn.CrossEntropyLoss()
# tudui =Tudui()
# optim=torch.optim.SGD(tudui.parameters(),lr=0.01)#设置优化器
#
# for epoch in range(20):
#     running_loss=0
#     for data in dataloader:
#         imgs,targets=data
#         output=tudui(imgs)#对其中的kernel进行优化
#         result_loss=loss(output,targets)
#         optim.zero_grad()#重置
#         result_loss.backward()#梯度
#         optim.step()#优化
#         running_loss=running_loss+result_loss
#     print(running_loss)
#
#
#
#
#
#
# input=torch.ones(64,3,32,32)
# output=tudui(input)
# print(output.shape)
#


