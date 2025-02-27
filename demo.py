import torch
from torch.utils.data import Dataset
import cv2
import os

# print(torch.cuda.is_available())
# print(dir(torch))
# print(dir(torch.cuda.is_available()))  # __XXX__不能去串改他
# print(help(torch.cuda.is_available))
print(help(Dataset))


class MYdata(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, item):
        return

    def __len__(self):
        return len(self.img_path)

