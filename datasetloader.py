from pathlib import Path
import glob

import torch
import os
from PIL import Image
import numpy as np
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader



class DriveTrainDataset(Dataset):
    def __init__(self,
                 root: str,
                 train: bool = True,
                 transforms = None):
        super(DriveTrainDataset, self).__init__()

        self.root = root
        self.flag = "train" if train else "test"

        dataroot = os.path.join(root, self.flag)
        assert os.path.exists(dataroot), f"path '{dataroot}' does not exists."

        img_names = [i for i in os.listdir(os.path.join(dataroot, "normal")) if i.endswith(".png")]

        self.normal_img_list = []
        self.surface_img_list = []
        self.mocusal_img_list = []
        self.tone_img_list = []
        for i in img_names:
            self.normal_img_list += [os.path.join(dataroot, "normal", i)]
            self.surface_img_list += [os.path.join(dataroot, "surface", i)]
            self.mocusal_img_list += [os.path.join(dataroot, "mocusal", i)]
            self.tone_img_list += [os.path.join(dataroot, "tone", i)]
        # print(img_names,"\n")

        self.mask = [os.path.join(dataroot, "label", i) for i in img_names]

        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.transforms = transforms

    def __getitem__(self, item):
        assert self.normal_img_list[item][-9:] == self.surface_img_list[item][-9:] == self.mocusal_img_list[item][-9:] == self.tone_img_list[item][-9:]
        normal_img = Image.open(self.normal_img_list[item])
        surface_img = Image.open(self.surface_img_list[item])
        mucosal_img = Image.open(self.mocusal_img_list[item])
        tone_img = Image.open(self.tone_img_list[item])

        mask = Image.open(self.mask[item]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            normal_img, surface_img, mucosal_img, tone_img, mask = self.transforms(normal_img, surface_img, mucosal_img, tone_img, mask)

        return normal_img, surface_img, mucosal_img, tone_img, mask

    def __len__(self):
        return len(self.normal_img_list)


class DriveTestDataset(Dataset):
    def __init__(self,
                 root: str,
                 transforms=None
                 ):
        super(DriveTestDataset, self).__init__()

        self.root = root
        dataroot = os.path.join(root, "test")

        img_names = [i for i in os.listdir(os.path.join(dataroot, "normal")) if i.endswith(".png")]

        self.normal_img_list = []
        self.surface_img_list = []
        self.mocusal_img_list = []
        self.tone_img_list = []
        for i in img_names:
            self.normal_img_list += [os.path.join(dataroot, "normal", i)]
            self.surface_img_list += [os.path.join(dataroot, "surface", i)]
            self.mocusal_img_list += [os.path.join(dataroot, "mocusal", i)]
            self.tone_img_list += [os.path.join(dataroot, "tone", i)]
        print(img_names, "\n")

        self.mask = [os.path.join(dataroot, "label", i) for i in img_names]

        for i in self.mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        self.transforms = transforms

    def __getitem__(self, item):
        normal_img = Image.open(self.normal_img_list[item])
        surface_img = Image.open(self.surface_img_list[item])
        mucosal_img = Image.open(self.mocusal_img_list[item])
        tone_img = Image.open(self.tone_img_list[item])

        mask = Image.open(self.mask[item]).convert('L')
        mask = np.array(mask) / 255
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            normal_img, surface_img, mucosal_img, tone_img, mask = self.transforms(normal_img, surface_img, mucosal_img, tone_img, mask)

        return normal_img, surface_img, mucosal_img, tone_img, mask

    def __len__(self):
        return len(self.normal_img_list)

if __name__ == "__main__":
    my_dataset = DriveTrainDataset(root="F:\dynasiam\\11", train=True)
    dl = DataLoader(
        my_dataset,
        batch_size=2
    )
    # my_test_dataset = DriveTestDataset(root="/home/kyasu/桌面/multimodal/datasets/normal")
    # dl = DataLoader(my_test_dataset, batch_size=1)
    ds = iter(dl)
    s = next(ds)    # normal: (B, D, H， W， C)



