# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys
import re
import cv2


def make_period_img(data, coordinate, img_shape=np.array([160, 160]), make_color=False, save_color=False):
    # coordinateは一行目がx, 二行目がy, make_colorは下限と上限を設定する（長さ2のリスト）
    coordinate = coordinate.astype(int)
    img = np.zeros((img_shape))
    print(coordinate[0])
    img[coordinate[1], coordinate[0]] = data
    if make_color is not False:
        hsv = np.ones((img_shape[0], img_shape[1], 3), dtype=np.float64) * 255

        hsv[coordinate[1], coordinate[0], 0] = (data - make_color[0]) / (make_color[1] - make_color[0]) * 150
        hsv[np.where(img < make_color[0])] = [0, 255, 255]
        hsv[np.where(img > make_color[1])] = [150, 255, 255]
        hsv[img == 0] = [0, 0, 0]
        hsv = hsv.astype(dtype=np.uint8)
        # print(hsv)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.imwrite(save_color + "_" + str(make_color[0]) + '_' + str(make_color[1]) + '_' + '.tif', bgr)
        return img, bgr
    else:
        return img


def make_period_imgs(imgs, make_color=[22, 28], save_color=False):
    #  make_colorは下限と上限を設定する（長さ2のリスト）
    hsv = np.ones((img_shape[0], img_shape[1], img_shape[2], 3), dtype=np.float64) * 255
    hsv[::, ::, ::, 0] = (imgs - make_color[0]) / (make_color[1] - make_color[0]) * 150
    hsv[np.where(imgs < make_color[0])] = [0, 255, 255]
    hsv[np.where(imgs > make_color[1])] = [150, 255, 255]
    hsv[imgs == 0] = [0, 0, 0]
    hsv = hsv.astype(dtype=np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if save_color is not False:
        cv2.imwrite(save_color + "_" + str(make_color[0]) + '_' + str(make_color[1]) + '_' + '.tif', bgr)
    return img, bgr


if __name__ == '__main__':
    # os.chdir('/media/kenya/HD-PLFU3/kenya/171013_jikan/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    os.chdir('/hdd1/kenya/Labo/keisan/python/_171019/result')
    # os.chdir('/media/kenya/HD-PLFU3/kenya/171013_jikan/sclipt/_171019/result')
    days = (['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX'])
    RAE = 'False'
    make_color = np.array([22, 28])
    # 解析データのフォルダ
    for day in days:
        for i in sorted(glob.glob(os.path.join(day, '*'))):
            for j in sorted(glob.glob(os.path.join(i, "fft", "*"))):
                if os.path.isdir(j):
                    data_file = os.path.join(j, "FFTnlls_results.csv")
                    print(data_file)
                    name_file = j.partition("data")[0] + 'name.csv'
                    print(name_file)
                    if os.path.isfile(data_file):
                        dataAll = np.loadtxt(data_file, delimiter=",", skiprows=1)
                        analysis = dataAll[:, 0]
                        data_amp = dataAll[analysis == 1, 1]
                        data_tau = dataAll[analysis == 1, 2]
                        data_phi = dataAll[analysis == 1, 3]
                        data_rae = dataAll[analysis == 1, 4]
                        nameAll = np.loadtxt(name_file, delimiter="\t")
                        coordinate = nameAll[0:2, analysis == 1]
                        make_period_img(data_tau, coordinate, img_shape=np.array([160, 160]), make_color=make_color, save_color=os.path.join(j, "RAE" + str(RAE)))
