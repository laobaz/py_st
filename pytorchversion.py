# 对视觉的处理
# 训练下载
import torchvision
from torch.utils.tensorboard import SummaryWriter

data_transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

train_Set = torchvision.datasets.CIFAR10(root="./dataset", transform=data_transform,train=True, download=True)
test_Set = torchvision.datasets.CIFAR10(root="./dataset",transform=data_transform, train=False, download=True)
# print(test_Set[0])
#
# img, target = test_Set[0]
# print(img)
# print(target)
# print(test_Set.classes[target])
# img.show()

print(test_Set[0])
writer=SummaryWriter("p10")
for i in range(10):
    img,target=test_Set[i]
    writer.add_image("test_set",img,i)

writer.close()

