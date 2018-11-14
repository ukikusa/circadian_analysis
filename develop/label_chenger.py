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


def label_change(imgs, change_frame):
    new_label = np.zeros_like(imgs)
    # change_frame = change_frame.fillna(0)
    change_frame = change_frame.dropna(how='all', axis=1)  # 不要な列を削除
    print(change_frame)
    label = np.array(change_frame.values).astype(np.int8)
    colum = np.array(change_frame.columns).astype(np.int8)
    print(label)
    for i, x in enumerate(colum):  # 置き換える値がx．それぞれの置き換える値に対して処理
        for j, y in enumerate(label[:, i]):  # 置き換えられる値がy それぞれの画像に対して処理
            if y != 0:  # 頭悪いのでおまじない．
                new_label[j, imgs[j] == y] = x
    return new_label


if __name__ == '__main__':
    # os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    # days = ('170613-LD2LL-ito-MVX')
    # days = ("./170829-LL2LL-ito-MVX")
    days = ("170215-LL2LL-MVX")
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    # days = ("./170215-LL2LL-MVX")
    data_folder = os.path.join(days, 'edit_raw')
    img_folder = os.path.join(data_folder, "label_imgs_tmp")  # 元ラベル画像
    change_file = os.path.join(data_folder, "label_180730" + ".csv")  # 変更に用いるファイル．

    # ここから処理
    imgs = im.read_imgs(img_folder, color=False)
    change = pd.read_csv(change_file, index_col=0, header=0)
    label_img = label_change(imgs, change)
    im.save_imgs(os.path.join(data_folder, 'label_img' + today), label_img)
