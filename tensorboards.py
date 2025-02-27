import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
writer = SummaryWriter("logs")
img_path="pytorch-tutorial-master/imgs/000.jpg"
img=cv2.imread(img_path)

writer.add_image("test",img,1,dataformats="HWC")
for i in range(100):
    writer.add_scalar("x=y",i,i)
writer.close()
