import os
from PIL import Image
import numpy as np
import pandas as pd
from tqdm import tqdm

def calculate_mean_and_std(img_h, img_w, img_c, image_path, label_path):
    # print("image size is: ", img_h, img_w, img_c)
    assert os.path.exists(image_path), f"image dir: '{image_path}' does not exist!"
    assert os.path.exists(label_path), f"image dir: '{label_path}' does not exist!"

    image_name_list = []
    for i in os.listdir(image_path):
        if i.endswith(".png"):
            image_name_list.append(i)
    cumulative_mean = np.zeros(img_c)
    cumulative_std = np.zeros(img_c)

    for img_name in tqdm(image_name_list):
        # print(img_name)
        img_path = os.path.join(image_path, img_name)
        msk_path = os.path.join(label_path, img_name)

        image = Image.open(img_path)
        image = image.resize((img_h, img_w))
        mask = Image.open(msk_path)
        mask = mask.resize((img_h, img_w))

        img = np.array(image) / 255
        msk_img = np.array(mask.convert('L'))

        img = img[msk_img == 255]

        if pd.isnull(img.mean(axis=0)).any() or pd.isnull(img.std(axis=0, ddof=1)).any():
            pass
        else:
            cumulative_mean += img.mean(axis=0)
            cumulative_std += img.std(axis=0, ddof=1)

    mean = cumulative_mean / len(image_name_list)
    std = cumulative_std / len(image_name_list)

    formatted_mean = "[" + ", ".join(f"{value:.4f}" for value in mean) + "]"
    formatted_std = "[" + ", ".join(f"{value:.4f}" for value in std) + "]"

    mean_values = formatted_mean.strip('[]').split(', ')
    std_values = formatted_std.strip('[]').split(', ')

    mean = [float(x) for x in mean_values]
    std = [float(x) for x in std_values]

    return mean, std
