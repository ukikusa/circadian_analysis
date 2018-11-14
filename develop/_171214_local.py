# -*-coding: utf-8 -*-

import numpy as np
import os
import itertools
import glob
import sys
import pandas as pd
import image_analysis as im
import cv2


def threshold(img, thresh=cv2.THRESH_OTSU, gaussian=False, kernel=5, local=False):  # local=nで(n,n)の範囲で二値化. ステップで枠をずらす幅
    # 現在実装済みのthreshは大津: cv2.THRESH_OTSU のみ．中央値は実装可能
    # 二値化の関数.
    # returnは8bit．threshold済み画像
    if np.max(img) > 255 or np.min(img) >= 10:
        img8 = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    if gaussian is not False:
        average = (gaussian, gaussian)  # Gaussaianで平滑化
        img8 = cv2.GaussianBlur(img8, average, 1)
    if local is not False:  # localに二値化する場合．
        img_threshold = np.empty_like(img8)
        thresh_n = np.empty_like(img8).astype(np.float16)
        l = int(local / 2)  # 範囲指定のため
        e = img8.shape[1] - local
        # 四隅
        thresh_n[:local, :local], img_threshold[:local, :local] = cv2.threshold(img8[:local, :local], 0, 255, cv2.THRESH_BINARY | thresh)
        thresh_n[:local, e:], img_threshold[:local, e:] = cv2.threshold(img8[:local, e:], 0, 255, cv2.THRESH_BINARY | thresh)
        thresh_n[e:, :local], img_threshold[e:, :local] = cv2.threshold(img8[e:, :local], 0, 255, cv2.THRESH_BINARY | thresh)
        thresh_n[e:, e:], img_threshold[e:, e:] = cv2.threshold(img8[e:, e:], 0, 255, cv2.THRESH_BINARY | thresh)
        for i in range(l + 1, img8.shape[0] - l - 1):  # 周辺
            # img_threshold[:local, i] = cv2.threshold(img8[:local, i - l:i + l + 1], 0, 255, cv2.THRESH_BINARY | thresh)[0][1][:, l]
            thresh_n[:local, i], tmp = cv2.threshold(img8[:local, i - l:i + l + 1], 0, 255, cv2.THRESH_BINARY | thresh)
            img_threshold[:local, i] = tmp[:, l]
            thresh_n[e:, i], tmp = cv2.threshold(img8[e:, i - l:i + l + 1], 0, 255, cv2.THRESH_BINARY | thresh)
            img_threshold[e:, i] = tmp[:, l]
            thresh_n[i, :local], tmp = cv2.threshold(img8[i - l:i + l + 1, :local], 0, 255, cv2.THRESH_BINARY | thresh)
            img_threshold[i, :local] = tmp[l, :]
            thresh_n[i, e:], tmp = cv2.threshold(img8[i - l:i + l + 1, e:], 0, 255, cv2.THRESH_BINARY | thresh)
            img_threshold[i, e:] = tmp[l, :]
        for i, j in itertools.product(range(l + 1, img8.shape[0] - l - 1), range(l + 1, img8.shape[1] - l - 1)):  # 内側
            thresh_n[i, j], tmp = cv2.threshold(img8[i - l:i + l + 1, j - l:j + l + 1], 0, 255, cv2.THRESH_BINARY | thresh)
            img_threshold[i, j] = tmp[l, l]
    else:
        thresh_n, img_threshold = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY | thresh)
    # 針を消すために縮小し，拡大する．
    if kernel is not False:
        ker_one = np.ones((kernel, kernel), np.uint8)
        img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_OPEN, ker_one)
        img_threshold = cv2.morphologyEx(img_threshold, cv2.MORPH_CLOSE, ker_one)
    return img_threshold, thresh_n


def threshold_imgs(imgs, thresh=cv2.THRESH_OTSU, gaussian=False, kernel=5, local=False):  # local=nで(n,n)の範囲で二値化.
    # 現在実装済みのthreshは大津: cv2.THRESH_OTSU のみ．中央値は実装可能
    # returnは8bit．threshold済み画像
    img_thresh = np.empty_like(imgs).astype(np.uint8)
    for i in range(imgs.shape[0]):
        img_thresh[i], thresh_n = threshold(imgs[i], thresh=thresh, gaussian=gaussian, kernel=kernel, local=local)
        print(i)
    return img_thresh, thresh_n


if __name__ == '__main__':
    # os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    days = ('./170215-LL2LL-MVX')
    data_folder = os.path.join(days, 'raw_data', 'data_light')
    imgs = im.read_imgs(data_folder, color=False)
    local, gaussian = 151, False
    img_thresh = threshold_imgs(imgs, thresh=cv2.THRESH_OTSU, gaussian=gaussian, local=local)
    im.save_imgs(os.path.join(days, 'edit_raw', 'thresh_local-' + str(local) + '_gaussian-' + str(gaussian)), img_thresh)
