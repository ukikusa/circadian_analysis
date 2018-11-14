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
    os.chdir(os.path.join('/hdd1','Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    days = (['170613-LD2LL-ito-MVX', "170829-LL2LL-ito-MVX", "170215-LL2LL-MVX"])
    kernel = np.ones((3,3),np.uint8)
    kernel[0,0] = 0
    kernel[0,2] = 0
    kernel[2,0] = 0
    kernel[2,2] = 0
    for day in days:
        div_folder = os.path.join(day, 'edit_raw', 'div_img')
        light_folder = os.path.join(day, 'raw_data', 'data_light')
        save_folder = os.path.join(day, 'edit_raw', 'div_tmp')
        light_imgs = im.read_imgs(light_folder, color=False)
        div_imgs = im.read_imgs(div_folder, color=False)
        div_imgs[div_imgs!=0] = 1
        dilate_img = np.empty_like(div_imgs)
        for j in range(div_imgs.shape[0]):
            dilate_img[j] = cv2.dilate(div_imgs[j],kernel,iterations = 1)
        edge = dilate_img - div_imgs
        light_imgs8 = im.bit1628(light_imgs)
        light_imgs8[edge==1]=255
        im.save_imgs(save_folder, light_imgs8)

    # ここから処理
    # mask_imgs = im.read_imgs(mask_folder, color=False)
    #  label_img = label_change_0_255(imgs, mask_imgs, change)
    # today = datetime.datetime.today().strftime("%Y-%m-%d")
    # im.save_imgs(os.path.join(data_folder, 'label_img-1-4-' + today), label_img)

    # label_img = label_change_4color(imgs, mask_imgs, change)
    # today = datetime.datetime.today().strftime("%Y-%m-%d")
    # im.save_imgs(os.path.join(data_folder, 'label_img-color4' + today), label_img, 'jpeg')
