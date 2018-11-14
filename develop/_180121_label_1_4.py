# -*-coding: utf-8 -*-

import numpy as np
import os
import itertools
import glob
import sys
import pandas as pd
import image_analysis as im
import cv2
import datetime


def label_change_0_255(imgs, mask_img, change_frame):
    new_label = np.zeros_like(imgs)
    new_label[mask_img != 0] = 155
    for i in change_frame:
        new_label[imgs == i] = 255
    return new_label

# BGR


def label_change_4color(imgs, mask_img, change_frame):
    new_label = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], 3), dtype=np.int16)
    new_label[mask_img == 255, :] = [125, 125, 125]
    # 黄色　水色　緑 赤
    color = np.array([[0, 255, 255], [255, 255, 0], [0, 128, 0], [0, 0, 255]])
    for i, j in zip(change_frame, color):
        new_label[imgs == i, :] = j

    return new_label


if __name__ == '__main__':
    # os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    days = ('./170613-LD2LL-ito-MVX')
    # days = ("./170829-LL2LL-ito-MVX")
    # days = ("./170215-LL2LL-MVX")
    data_folder = os.path.join(days, 'edit_raw')
    img_folder = os.path.join(data_folder, "label_img")
    mask_folder = os.path.join(data_folder, "div_frond")
    change = np.array([1, 2, 3, 4])

    # ここから処理
    imgs = im.read_imgs(img_folder, color=False)
    mask_imgs = im.read_imgs(mask_folder, color=False)
    label_img = label_change_0_255(imgs, mask_imgs, change)
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    im.save_imgs(os.path.join(data_folder, 'label_img-1-4-' + today), label_img)

    label_img = label_change_4color(imgs, mask_imgs, change)
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    im.save_imgs(os.path.join(data_folder, 'label_img-color4' + today), label_img, 'jpeg')
