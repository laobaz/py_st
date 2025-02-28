import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from CIFAR_10 import *

dataset_train = torchvision.datasets.CIFAR10(root="./dataset", train=True, transform=torchvision.transforms.ToTensor(),
                                             download=False)
dataset_test = torchvision.datasets.CIFAR10(root="./dataset", train=False, transform=torchvision.transforms.ToTensor(),
                                            download=False)

train_dataloader = DataLoader(dataset_train, batch_size=64)
test_dataloader = DataLoader(dataset_test, batch_size=64)

print("train is num{}".format((len(dataset_train))))
print("test is num{}".format((len(dataset_test))))
# tudui = Tudui()
tudui =torch.load("./model/tudui_10.pth",weights_only=False)
tudui.cuda()

loss_fn = nn.CrossEntropyLoss()

loss_fn=loss_fn.cuda()
learning_rate = 1e-2
optim = torch.optim.SGD(tudui.parameters(), learning_rate)

total_train_step = 0

total_test_step = 0
epoch = 10

writer = SummaryWriter("demo1")
for i in range(epoch):

    tudui.train()  # 只对部分模型层有用
    print("第 {} 轮开始了".format(i + 1))
    total_train_step = 0
    # 训练器开始
    for data in train_dataloader:
        imgs, targets = data
        imgs=imgs.cuda()
        targets=targets.cuda()
        output = tudui(imgs)
        loss = loss_fn(output, targets)
        # 优化器调优
        optim.zero_grad()
        loss.backward()
        optim.step()
        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{} ,loss:{}".format(total_train_step, loss))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    tudui.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 这里面的不会被调优
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = tudui(imgs)
            loss = loss_fn(output, targets)
            total_test_loss = total_test_loss + loss
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = accuracy + total_accuracy
    print("测试集上的loss{}".format(total_test_loss))
    total_test_step = total_test_step + 1
    writer.add_scalar("test_loss:", total_test_loss.item(), total_test_step)
    print("测试集上的准确率{}".format(total_accuracy/len(dataset_test)))
    torch.save(tudui, "./model/tudui_{}.pth".format(total_test_step))
    print("模型已保存")

writer.close()
