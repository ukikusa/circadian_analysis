# -*-coding: utf-8 -*-

import numpy as np
import os
import itertools
import glob
import sys
import pandas as pd
import image_analysis as im
import cv2


if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    days = (['170613-LD2LL-ito-MVX', "170829-LL2LL-ito-MVX", "170215-LL2LL-MVX"])
    kernel = np.ones((3, 3), np.uint8)
    # kernel[0,0] = 0
    # kernel[0,2] = 0
    # kernel[2,0] = 0
    # kernel[2,2] = 0
    div_folder = os.path.join('170613-LD2LL-ito-MVX', 'edit_raw', 'div_tmp_01')
    save_folder = os.path.join('170613-LD2LL-ito-MVX', 'edit_raw', 'div_tmp_01_3')
    div_imgs = im.read_imgs(div_folder, color=False)
    div_imgs[div_imgs != 0] = 1
    dilate_img = np.empty_like(div_imgs)
    for j in range(div_imgs.shape[0]):
        edge = cv2.morphologyEx(div_imgs[j], cv2.MORPH_CLOSE, kernel)
        edge = edge - div_imgs[j]
        print(np.sum(edge))
        dilate_img[j] = cv2.dilate(edge, kernel, iterations=1)
        print(np.sum(dilate_img[j]))
    div_imgs[dilate_img == 1] = 0
    div_imgs[div_imgs == 1] = 255
    im.save_imgs(save_folder, div_imgs)

    # ここから処理
    # mask_imgs = im.read_imgs(mask_folder, color=False)
    #  label_img = label_change_0_255(imgs, mask_imgs, change)
    # today = datetime.datetime.today().strftime("%Y-%m-%d")
    # im.save_imgs(os.path.join(data_folder, 'label_img-1-4-' + today), label_img)

    # label_img = label_change_4color(imgs, mask_imgs, change)
    # today = datetime.datetime.today().strftime("%Y-%m-%d")
    # im.save_imgs(os.path.join(data_folder, 'label_img-color4' + today), label_img, 'jpeg')
