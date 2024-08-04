import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import functional as F
import os
from tqdm import tqdm
import cv2

def Texture_Enhance(image_path, image_name, texture_enhance_save_path):
    path = os.path.join(image_path, image_name)
    image = cv2.imread(os.path.join(image_path, image_name), 1)

    # 划分为三个颜色平面
    b, g, r = cv2.split(image)
    h, w = r.shape

    # 标准化
    min_r, max_r = np.min(r), np.max(r)
    min_g, max_g = np.min(g), np.max(g)
    min_b, max_b = np.min(b), np.max(b)
    r = (r - min_r) / max_r
    g = (g - min_g) / max_g
    b = (b - min_b) / max_b

    # 求模糊图
    r_tmp = cv2.GaussianBlur(r, (5, 5), 0, 0)
    g_tmp = cv2.GaussianBlur(g, (5, 5), 0, 0)
    b_tmp = cv2.GaussianBlur(b, (5, 5), 0, 0)

    # 包含图像中的高频成分
    r_eg = r - r_tmp
    g_eg = g - g_tmp
    b_eg = b - b_tmp

    t1 = 5 * (np.sum(r) / (h * w))
    t2 = 5 * (np.sum(g) / (h * w))
    t3 = 5 * (np.sum(b) / (h * w))

    r_sharpen = r + t1 * r_eg
    g_sharpen = g + t2 * g_eg
    b_sharpen = b + t3 * b_eg

    # 重建RGB图像并输出到屏幕
    b, g, r = np.uint8(255 * b_sharpen), np.uint8(255 * g_sharpen), np.uint8(255 * r_sharpen)
    image_te = cv2.merge([b, g, r])

    cv2.imwrite(os.path.join(texture_enhance_save_path, image_name), image_te)

def Detail_Enhance(image_path, image_name, detail_enhance_save_path):
    image = cv2.imread(os.path.join(image_path, image_name), 1)

    b, g, r = cv2.split(image)
    h, w = b.shape

    min_r, max_r = np.min(r), np.max(r)
    min_g, max_g = np.min(g), np.max(g)
    min_b, max_b = np.min(b), np.max(b)

    r = (r - min_r) / max_r
    g = (g - min_g) / max_g
    b = (b - min_b) / max_b

    r_tmp = cv2.GaussianBlur(r, (5, 5), 0, 0)
    g_tmp = cv2.GaussianBlur(g, (5, 5), 0, 0)
    b_tmp = cv2.GaussianBlur(b, (5, 5), 0, 0)

    # 锐化
    r_eg = r - r_tmp
    g_eg = g - g_tmp
    b_eg = b - b_tmp

    t1 = 5 * (np.sum(r) / (h * w))
    t2 = 5 * (np.sum(g) / (h * w))
    t3 = 5 * (np.sum(b) / (h * w))

    r_sharpen = r + t1 * r_eg
    g_sharpen = g + t2 * g_eg
    b_sharpen = b + t3 * b_eg

    # R 通道
    tmp_r = np.sum(r_sharpen) / (h * w)
    k = 0.2 + np.power(0.5, tmp_r)
    g = 10 * np.log(23 / 10) * tmp_r
    tmp_r = 1 / (1 + np.exp(g * (k - r_sharpen)))
    tmp_r = np.uint8(255 * tmp_r)

    # B 通道
    b_sharpen = np.uint8(255 * b_sharpen)
    # m_b = (0.06 + np.log(10)) * (np.sum(b_sharpen) / (h * w))
    mask_b = cv2.adaptiveThreshold(b_sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
    # _, mask_b = cv2.threshold(b_sharpen, m_b, 255, cv2.THRESH_BINARY)
    tmp_b = cv2.inpaint(b_sharpen, mask_b, 10, cv2.INPAINT_TELEA)

    # G 通道
    g_sharpen = np.uint8(255 * g_sharpen)
    max_percentile_pixel = np.percentile(g_sharpen, 99)
    min_percentile_pixel = np.percentile(g_sharpen, 1)
    g_sharpen[g_sharpen >= max_percentile_pixel] = max_percentile_pixel
    g_sharpen[g_sharpen <= min_percentile_pixel] = min_percentile_pixel
    tmp_g = np.zeros(g_sharpen.shape, g_sharpen.dtype)
    cv2.normalize(g_sharpen, tmp_g, 0, 255, cv2.NORM_MINMAX)

    b, g, r = tmp_b, tmp_g, tmp_r
    image_de = cv2.merge([b, g, r])

    cv2.imwrite(os.path.join(detail_enhance_save_path, image_name), image_de)

def Color_Enhance(image_path, image_name, color_enhance_save_path):
    image = cv2.imread(os.path.join(image_path, image_name), 1)

    # 划分为三个颜色平面
    b, g, r = cv2.split(image)
    h, w = r.shape

    # 标准化
    min_r, max_r = np.min(r), np.max(r)
    min_g, max_g = np.min(g), np.max(g)
    min_b, max_b = np.min(b), np.max(b)

    r = (r - min_r) / max_r
    g = (g - min_g) / max_g
    b = (b - min_b) / max_b

    # 求模糊版本
    r_tmp = cv2.GaussianBlur(r, (5, 5), 0, 0)
    g_tmp = cv2.GaussianBlur(g, (5, 5), 0, 0)
    b_tmp = cv2.GaussianBlur(b, (5, 5), 0, 0)

    # 锐化
    r_eg = r - r_tmp
    g_eg = g - g_tmp
    b_eg = b - b_tmp

    t1 = 10 * (np.sum(r) / (h * w))
    t2 = 10 * (np.sum(g) / (h * w))
    t3 = 10 * (np.sum(b) / (h * w))

    r_sharpen = r + t1 * r_eg
    g_sharpen = g + t2 * g_eg
    b_sharpen = b + t3 * b_eg

    b, g, r = np.uint8(255 * b_sharpen), np.uint8(255 * g_sharpen), np.uint8(255 * r_sharpen)

    # 统计每个像素值对应多少像素点，并每个像素值对应的直方图值
    r_n = np.bincount(r.reshape(-1), minlength=256)
    r_p = r_n / (h * w)
    g_n = np.bincount(g.reshape(-1), minlength=256)
    g_p = g_n / (h * w)
    b_n = np.bincount(b.reshape(-1), minlength=256)
    b_p = b_n / (h * w)

    # 映射函数
    from collections import defaultdict
    r_new_hist = defaultdict(lambda: 0)
    for i in range(0, 256):
        for pixel in range(0, i + 1):
            r_new_hist[i] += r_p[pixel]
        r_new_hist[i] = r_new_hist[i] * 255
        r_new_hist[i] = np.around(r_new_hist[i])
    r_cte = r.copy()
    for i in range(r_cte.shape[0]):
        for j in range(r_cte.shape[1]):
            r_cte[i, j] = r_new_hist[r_cte[i, j]]

    g_new_hist = defaultdict(lambda: 0)
    for i in range(0, 256):
        for pixel in range(0, i + 1):
            g_new_hist[i] += g_p[pixel]
        g_new_hist[i] = g_new_hist[i] * 255
        g_new_hist[i] = np.around(g_new_hist[i])
    g_cte = g.copy()
    for i in range(g_cte.shape[0]):
        for j in range(g_cte.shape[1]):
            g_cte[i, j] = g_new_hist[g_cte[i, j]]

    b_new_hist = defaultdict(lambda: 0)
    for i in range(0, 256):
        for pixel in range(0, i + 1):
            b_new_hist[i] += b_p[pixel]
        b_new_hist[i] = b_new_hist[i] * 255
        b_new_hist[i] = np.around(b_new_hist[i])
    b_cte = b.copy()
    for i in range(b_cte.shape[0]):
        for j in range(b_cte.shape[1]):
            b_cte[i, j] = b_new_hist[b_cte[i, j]]

    image_ce = cv2.merge([b_cte, g_cte, r_cte])

    cv2.imwrite(os.path.join(color_enhance_save_path, image_name), image_ce)


class TDC_Enhance(object):
    def __init__(self, image_path, texture_enhance_save_path, detail_enhance_save_path, color_enhance_save_path):
        self.image_path = image_path
        self.texture_enhance_save_path = texture_enhance_save_path
        self.detail_enhance_save_path = detail_enhance_save_path
        self.color_enhance_save_path = color_enhance_save_path

    def __call__(self):
        for image_name in tqdm(os.listdir(self.image_path)):
            if image_name.endswith('.png'):
                Texture_Enhance(self.image_path, image_name, self.texture_enhance_save_path)
                Detail_Enhance(self.image_path, image_name, self.detail_enhance_save_path)
                Color_Enhance(self.image_path, image_name, self.color_enhance_save_path)


