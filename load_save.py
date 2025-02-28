


import torch
import torchvision.models


#保存方法一：加载模型
vgg16=torchvision.models.vgg16(weights=None)
torch.save(vgg16,"vgg16_mothod1.pth")

#方法二：模型参数保存
torch.save(vgg16.state_dict(),"vgg16_method2.pth")

#加载权重
vgg16.load_state_dict(torch.load('vgg16_method2.pth'))


vgg1=torch.load("vgg16_mothod1.pth",weights_only=False)
print(vgg1)

vgg2=torch.load("vgg16_method2.pth",weights_only=False)
print(vgg2)