# -*- coding: utf-8 -*-

import image_analysis as im
import os
import glob
import numpy as np
import sys
import itertools

def img_for_bio(img):
    # 移動補正が最も長く取れた画像を使う
    label = np.amax(img, axis=(1,2))
    # 枚数の多いラベルを返す．uniqueで[[値]，[数]]
    label_n = np.unique(label[np.nonzero(label)], return_counts=True)
    img[img!=label_n[0][np.argmax(label_n[1])]] = 0
    img[img.nonzero()] = 255
    # 差分を取る．これにより0から255で255，255から0で1が返される
    img_diff = np.diff(img, axis=0)
    # 無駄なループ回したくないのでフロンドがあるところだけでループ
    pixel = np.nonzero(np.max(img, axis=0))
    print(pixel[0])
    small_img = np.zeros_like(img)  # 箱を作る
    for i, j in zip(pixel[0],pixel[1]):
        img_i = np.zeros_like(img[:, i, j])
        diff_255, diff_1 = np.where(img_diff[:, i, j] == 255)[0], np.where(img_diff[:, i, j] == 1)[0]
        if np.size(diff_255) + np.size(diff_1) <= 2 :  # 発光が見えてる期間は一連である OK
            img_i = img[:, i, j]
        elif np.logical_and(np.size(diff_255) == np.size(diff_1), diff_255[0] < diff_1[0]):  # 前も後ろも消えてる OK
            x = np.argmax(diff_1 - diff_255)
            img_i[diff_255[x]+1:diff_1[x]+1] = 255  # img[diff_255[x]+1:diff_1[x]+1, i]
        elif np.logical_and(np.size(diff_255) == np.size(diff_1), diff_255[0] > diff_1[0]):  # 前も後ろも消えてない
            x = np.argmax(diff_1[1:] - diff_255[:-1])
            x_max = np.max(diff_1[1:] - diff_255[:-1])
            if np.argmax([diff_1[0], x_max, img.shape[0]-diff_255[-1]]) == 0:  # 前が長い OK
                img_i[:diff_1[0]+1] = 255
            elif np.argmax([diff_1[0], x_max, img.shape[0]-diff_255[-1]]) == 1:  # 真ん中が長い OK
                img_i[diff_255[x] + 1:diff_1[x+1] + 1] = 255
            else:  # ケツが長い OK
                img_i[diff_255[-1]+1:] = 255
        elif diff_255[0] > diff_1[0]:  # 前にフロンドがない
            x = np.argmax(diff_1[1:] - diff_255)
            if diff_1[0] > np.max(diff_1[1:] - diff_255):
                img_i[:diff_1[0]+1] = 255
            else:
                img_i[diff_255[x] + 1:diff_1[x+1] + 1] = 255
        else: #後ろにフロンドがない
            x = np.argmax(diff_255[:-1]-diff_1)
            if (img.shape[0]-diff_255[-1]) > np.max(diff_1 - diff_255[:-1]):
                img_i[diff_255[-1]+1:] = 255
            else:
                img_i[diff_255[x] + 1:diff_1[x] + 1] = 255
        small_img[:, i, j] = img_i
    return small_img

if __name__ == '__main__':
    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    # 処理したいデータのフォルダ
    # day = ('./170829-LL2LL-ito-MVX')
    days = ['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX']
    # 解析データのフォルダ
    for day in days:
        frond_folder = day + '/frond'
        for i in sorted(glob.glob(frond_folder + '/*')):
            print(i)
            img = im.read_imgs(i + '/moved_mask_frond')
            lum_img = im.read_imgs(i + '/moved_mask_frond_lum')
            small_img = img_for_bio(img)
            im.save_imgs(i + '/small_moved_mask_frond', small_img)
            lum_img[small_img == 0] = 0
            im.save_imgs(i + '/small_moved_mask_frond_lum', lum_img)
