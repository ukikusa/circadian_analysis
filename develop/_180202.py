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


def writes_text(img, text, org, fontScale, color=255):
    # org – 文字列の左下角の，画像中の座標
    # fontFace – フォントの種類．以下のうちの1つ． FONT_HERSHEY_SIMPLEX , FONT_HERSHEY_PLAIN , FONT_HERSHEY_DUPLEX , FONT_HERSHEY_COMPLEX , FONT_HERSHEY_TRIPLEX , FONT_HERSHEY_COMPLEX_SMALL , FONT_HERSHEY_SCRIPT_SIMPLEX , FONT_HERSHEY_SCRIPT_COMPLEX ．また，各フォントIDを， FONT_HERSHEY_ITALIC と組み合わせて，斜体文字
    # fontScale – フォントのスケールファクタ．これがフォント特有の基本サイズに掛け合わされます
    # lineType – 線の種類．詳細は line を参照してください．
    fontFace = cv2.FONT_HERSHEY_SIMPLE
    cv2.putText(img, text, org, fontFace, fontScale=fontScale, color=color, thickness=1)
    return img


def wirte_text(imgs, day=True, offset=0, color=255):
    text1 = 'Day ' + str(1)
    text2 = '00:00'
    left = int(0.02 * imgs.shape[1])
    top = int(0.06 * imgs.shape[2])
    fontScale = imgs.shape[1] / 500
    for i in range(imgs.shape[0]):
        day = 'Day ' + str(int((i + offset) / 24))
        time = str(np.mod(i + int(offset), 24)) + ':' + str(int(np.modf(offset)[1] * 60))
        imgs[i] = writes_text(imgs[i], day, (left, top), fontScale, color=color)
        imgs[i] = writes_text(imgs[i], time, (left, top * 2), fontScale, color=color)
    return 0


if __name__ == '__main__':
    # os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    days = ('./170613-LD2LL-ito-MVX')
    # days = ("./170829-LL2LL-ito-MVX")
    # days = ("./170215-LL2LL-MVX")
    data_folder = os.path.join(days, 'edit_raw')
    img_folder = os.path.join(data_folder, "lum_min_img")
    img_folder = os.path.join(days, "frond", 'label-001_239-188_n214', 'frond')
    imgs = im.read_imgs(img_folder, color=False)
    imgs = im.bit1628(imgs)

    cv2.imwrite('tmp.jpg', img)

    # ここから処理
    # mask_imgs = im.read_imgs(mask_folder, color=False)
    #  label_img = label_change_0_255(imgs, mask_imgs, change)
    # today = datetime.datetime.today().strftime("%Y-%m-%d")
    # im.save_imgs(os.path.join(data_folder, 'label_img-1-4-' + today), label_img)

    # label_img = label_change_4color(imgs, mask_imgs, change)
    # today = datetime.datetime.today().strftime("%Y-%m-%d")
    # im.save_imgs(os.path.join(data_folder, 'label_img-color4' + today), label_img, 'jpeg')
