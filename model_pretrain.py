import torchvision
from torch import nn

import transform

# train_data=torchvision.datasets.ImageNet("../data_image_net",split="train",download=True,
#                                          transform=torchvision.transforms.ToTensor())


vgg16_false=torchvision.models.vgg16(pretained=False)
vgg16_True=torchvision.models.vgg16(pretained=True)#有预训练


vgg16_True.classifier.add_module("add_linear",nn.Linear(1000,10))


print(vgg16_True)