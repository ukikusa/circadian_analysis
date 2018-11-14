# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys
import image_analysis as im
import cv2
import datetime
from _171214_local import threshold_imgs
from _171214_local import threshold

if __name__ == '__main__':
    os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    # days = ('./170613-LD2LL-ito-MVX')
    days = ("./170215-LL2LL-MVX")
    data_folder = os.path.join(days, 'raw_data')
    if 1==10:
        img_folder = os.path.join(data_folder, "data_light")
        save_file = os.path.join(data_folder,"data_min.tif")

        ##### ここから処理
        imgs = im.read_imgs(img_folder, color=False)
        img = np.max(imgs, axis = 0)
        print(img.shape)
        cv2.imwrite(save_file, img)


    img_folder = os.path.join(data_folder, "data_light")
    img_max_f = os.path.join(data_folder,"data_max_edit.tif")
    save_folder = os.path.join(days,"edit_raw", "data_max")
    
    ##### ここから処理
    imgs = im.read_imgs(img_folder, color=False)
    img_max = cv2.imread(img_max_f, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
    img_max_n = np.max(imgs, axis = 0)
    img_max[img_max<img_max_n] = img_max_n[img_max<img_max_n]
    imgs_max = (-imgs + img_max).astype(np.uint16)
    # im.save_imgs(save_folder, imgs_max)
    imgs_thresh = np.zeros_like(imgs_max) #.astype(np.uint8)
    imgs_max8 = im.bit1628(imgs_max)
    imgs8 = im.bit1628(imgs)
    # img_thresh, tmp = threshold(imgs8[-1], thresh=cv2.THRESH_OTSU, gaussian=3, local=75)
    # print(img_thresh, tmp)
    # imgs_thresh[imgs8>tmp] = imgs8[imgs8>tmp]
    imgs_thresh, tmp = threshold_imgs(imgs_max, thresh=cv2.THRESH_OTSU, gaussian=3, kernel=7, local=False)
    im.save_imgs(os.path.join(days,"edit_raw", "thresh"), imgs_thresh)
