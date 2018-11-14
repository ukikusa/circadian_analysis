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
    days = (['170613-LD2LL-ito-MVX'])  # , "170829-LL2LL-ito-MVX", "170215-LL2LL-MVX"])
    kernel = np.ones((3, 3), np.uint8)
    kernel[0, 0] = 0
    kernel[0, 2] = 0
    kernel[2, 0] = 0
    kernel[2, 2] = 0
    for day in days:
        light_folder = os.path.join(day, 'raw_data', 'data_light')
        edit_folder = os.path.join(day, 'edit_raw', 'tmp', '03')
        div_folder = os.path.join(day, 'edit_raw', 'div_img')
        save_folder = os.path.join(day, 'edit_raw', 'div_tmp_01')
        light_imgs = im.read_imgs(light_folder, color=False)
        div_imgs = im.read_imgs(div_folder, color=False)
        edit_imgs = im.read_imgs(edit_folder, color=False)
        edit_imgs[edit_imgs == 71] = 255
        new_div = np.ones_like(div_imgs)
        new_div[edit_imgs==255] = 0
        for i in range(new_div.shape[0]):
            label_n, label_img = cv2.connectedComponents(new_div[i], connectivity=4)
            for j in range(label_n):
                new_div[i][label_img == j] = np.argmax(np.bincount(div_imgs[i][label_img == j]))
        im.save_imgs(save_folder, new_div)
