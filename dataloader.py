from torch.utils.data import DataLoader
import torchvision
from torch.utils.tensorboard import SummaryWriter

test_Set = torchvision.datasets.CIFAR10("./dataset", False, transform=torchvision.transforms.ToTensor())
test_laoder = DataLoader(dataset=test_Set, batch_size=4, shuffle=True, num_workers=0)

img, target = test_Set[0]
writer=SummaryWriter("dataloader")
step=0
for data in test_laoder:
    imgs, target = data
    writer.add_images("datalaoder:{}".format(step),imgs,step)
    step=step+1
writer.close()

