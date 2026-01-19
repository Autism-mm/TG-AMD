import torchvision.datasets as datasets
from torchvision import transforms
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
def load_data(file_dir, transform):
    data_set = datasets.ImageFolder(file_dir, transform=transform)
    num_classes = len(data_set.classes)
    return data_set,num_classes
#
# data_transform = transforms.Compose([
#     # transforms.Resize(128), # 缩放到 96 * 96 大小
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
#灰度标准化
# data_transform = transforms.Compose([
#     transforms.Resize((128,128)), # 缩放到 128 * 128 大小
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])
# data_transform = transforms.Compose([
#     transforms.Resize(224),  # 缩放到 96 * 96 大小
#     transforms.ToTensor(),
#     # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])
data_transform = transforms.Compose([
    # transforms.Resize(224), # 缩放到 96 * 96 大小
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class custom_dset(Dataset):
    def __init__(self,
                 img_path,
                 txt_path,
                 img_transform=None,):
        self.img_list =img_path
        self.label_list = txt_path
        self.img_transform = img_transform

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)  # 可以读取单通道影像,读取3通道16位tif影像时报错(PIL.UnidentifiedImageError: cannot identify image file),支持4通道8位影像

        label = self.label_list[index]
        label = torch.Tensor([label]).type(torch.LongTensor).squeeze()
        # img = self.loader(img_path)
        img = np.expand_dims(img,axis=2)
        img = np.concatenate([img, img, img], axis=2)
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, label

    def __len__(self):
        return len(self.label_list)