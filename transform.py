from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from PIL import Image
img_path="pytorch-tutorial-master/imgs/002.jpg"
img=Image.open(img_path)

tensort=transforms.ToTensor()
tensort_t=tensort(img)
print(tensort_t[0][0][0])

writer = SummaryWriter("logs")
writer.add_image("demo",tensort_t,1)

#Normalise 归一化

trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(tensort_t)
print(img_norm[0][0][0])
writer.add_image("Demo1",img_norm,2)


#resize
print(img.size)
trans_resize=transforms.Resize((512,512))
img_resize=trans_resize(img)#只变了大小
img_resize=tensort(img_resize)#变了type
writer.add_image("demo2",img_resize,3)

trans_resize_2=transforms.Resize(512)
trans_compose=transforms.Compose(trans_resize_2,tensort)
img_resize_2=trans_compose(img)


writer.close()
