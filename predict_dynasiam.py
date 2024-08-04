import glob
import os
import time
import cv2
from pathlib import Path

import matplotlib.pyplot as plt
import torch.nn.functional as F

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from dynasiam.src.DynaSiam import mmNet


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    weights_path = "your path"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    normal_root = "your path"
    surface_root = "your path"
    mucosal_root = "your path"
    tone_root = "your path"
    assert os.path.exists(normal_root), f"weights {normal_root} not found."
    assert os.path.exists(surface_root), f"weights {surface_root} not found."
    assert os.path.exists(mucosal_root), f"weights {mucosal_root} not found."
    assert os.path.exists(tone_root), f"weights {tone_root} not found."

    device = "cuda"
    print("using {} device.".format(device))

    normal_img = Image.open(normal_root).convert('RGB')
    surface_img = Image.open(surface_root).convert('RGB')
    mucosal_img = Image.open(mucosal_root).convert('RGB')
    tone_img = Image.open(tone_root).convert('RGB')

    model = mmNet(is_training=False)
    print("creating model success")

    data_transform_normal = transforms.Compose([transforms.Resize((224, 224)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                     std=(0.5, 0.5, 0.5))])
    img_normal = data_transform_normal(normal_img)
    img_normal = torch.unsqueeze(img_normal, dim=0)

    data_transform_surface = transforms.Compose([transforms.Resize((224, 224)),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                      std=(0.5, 0.5, 0.5))])
    img_surface = data_transform_surface(surface_img)
    img_surface = torch.unsqueeze(img_surface, dim=0)

    data_transform_muccosal = transforms.Compose([transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                       std=(0.5, 0.5, 0.5))])
    img_mucosal = data_transform_muccosal(mucosal_img)
    img_mucosal = torch.unsqueeze(img_mucosal, dim=0)


    data_transform_tone = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                                                   std=(0.5, 0.5, 0.5))])
    img_tone = data_transform_tone(tone_img)
    img_tone = torch.unsqueeze(img_tone, dim=0)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location=device)['model'], strict=False)
    model.to(device)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        img_normal = img_normal.cuda()
        img_surface = img_surface.cuda()
        img_mucosal = img_mucosal.cuda()
        img_tong = img_tone.cuda()
        output = model(img_normal, img_surface, img_mucosal, img_tong)

        prediction = output.argmax(1).squeeze(0)
        # prediction = output[0].squeeze(0).argmax(0)

        mask = prediction.to('cpu').numpy().astype(np.float64)
        mask[mask > 0.9] = 255
        mask[mask == 0] = 0

        # plt.imshow(mask, cmap=plt.cm.gray)
        # plt.show()

        mask = Image.fromarray(mask)
        if mask.mode == "F":
            mask = mask.convert("RGB")
        mask.save("")


if __name__ == '__main__':
    main()
