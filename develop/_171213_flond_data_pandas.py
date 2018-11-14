# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys
import datetime
import pandas as pd
import image_analysis as im


def frond_folder2data(folder, time=60, offset=0, save=True):
    folder_list = sorted(os.listdir(folder))
    day = os.path.split(folder)[0]
    print(folder_list)
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    img_number = len(glob.glob(os.path.join(folder, folder_list[0], 'mask_frond_lum', '*.tif')))
    time_list = np.linspace(offset, (img_number - 1) * time / 60, img_number, dtype=np.float)
    frond_lum = np.empty((img_number, len(folder_list)), dtype=np.uint32)
    frond_area = np.empty((img_number, len(folder_list)), dtype=np.uint16)
    frond_avg = np.empty_like(frond_area)
    for (i, s) in enumerate(folder_list):
        print(s)
        img = im.read_imgs(os.path.join(folder, s, 'mask_frond_lum'))
        frond_lum[:, i] = np.sum(img, axis=(1, 2))
        frond_area[:, i] = np.count_nonzero(img, axis=(1, 2))
    frond_avg = frond_lum / frond_area
    frond_lum, frond_avg, frond_area = pd.DataFrame(frond_lum), pd.DataFrame(frond_avg), pd.DataFrame(frond_area)
    frond_lum.columns, frond_area.columns, frond_avg.columns = folder_list, folder_list, folder_list
    frond_lum.index, frond_area.index, frond_avg.index = time_list, time_list, time_list
    if os.path.exists(os.path.join(day, 'result')) is False:
        os.mkdir(os.path.join(day, 'result'))
    if save is True:
        frond_lum.to_csv(os.path.join(day, 'result', today + 'frond_lum_sum.csv'))
        frond_avg.to_csv(os.path.join(day, 'result', today + 'frond_lum_avg.csv'))
        frond_area.to_csv(os.path.join(day, 'result', today + 'frond_area.csv'))
    return frond_lum, frond_avg, frond_area


if __name__ == '__main__':
    #    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    # days = ('./170613-LD2LL-ito-MVX')
    day = ("./170829-LL2LL-ito-MVX")
    time = 60
    offset = 0
    # flondの発光量，面積，平均を求める．
    folder = os.path.join(day, 'frond')
    frond_folder2data(folder, time=time, offset=offset, save=True)
