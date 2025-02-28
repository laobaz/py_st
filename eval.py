import torch
import torchvision.transforms
from PIL import Image

image_path = "./img/dog.png"

img = Image.open(image_path)
print(img)
# 修改图片大小
# 先变大小，再换格式 3x32x32
transform = torchvision.transforms.Compose([torchvision.transforms.Resize((32, 32)),
                                            torchvision.transforms.ToTensor()])

img = transform(img)
print(img.shape)


tudui = torch.load("./model/tudui_10.pth", weights_only=False)

# tudui为四维，但是img之后三维，没有batch_size，需要添加一维
img = torch.reshape(img, (-1, 3, 32, 32))

tudui.eval()
with torch.no_grad():
    #因为使用gpu加速了，所以img也要用gpu
    img=img.cuda()
    output = tudui(img)
print(output)

#得出相对的结果
print(output.argmax(1))
