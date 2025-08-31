import os
import torch
from torch.utils.data import  Dataset
from PIL import Image
import random

class DataLoad(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform

        random.seed(0)
        assert os.path.exists(data_path), "dataset root: {} does not exist.".format(data_path)

        dataset_class = [cla for cla in os.listdir(os.path.join(data_path))]
        self.num_class = len(dataset_class)

        dataset_class.sort()

        class_indices = dict((cla, idx) for idx, cla in enumerate(dataset_class))

        self.images_path = []
        self.images_label = []
        self.images_num = []
        supported = [".jpg", ".JPG", ".png", ".PNG"]

        for cla in dataset_class:
            cla_path = os.path.join(data_path, cla)

            images = [os.path.join(data_path, cla, i) for i in os.listdir(cla_path) if
                      os.path.splitext(i)[-1] in supported]

            image_class = class_indices[cla]

            self.images_num.append(len(images))

            for img_path in images:
                self.images_path.append(img_path)
                self.images_label.append(image_class)

        print("{} images were found in the dataset.".format(sum(self.images_num)))

    def __len__(self):
        return sum(self.images_num)

    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        label = self.images_label[idx]
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
        if self.transform is not None:
            img = self.transform(img)
        else:
            raise ValueError('Image is not preprocessed')
        return img, label


    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
