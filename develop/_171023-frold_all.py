# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


if __name__ == '__main__':
    # os.chdir('/media/kenya/HD-PLFU3/kenya/171013_jikan/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    os.chdir('/hdd1/Users/kenya/gojira/171023_jikan/_171019/result')
    days = (['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX'])
    # 解析データのフォルダ

    for day in days:
        folders = sorted(glob.glob(os.path.join(day, '*')))
        for i, folder in enumerate(folders):
            print(folder)
            data = np.loadtxt(os.path.join(folder, 'raw_data.csv'), delimiter='\t')
            # とりあえず座標の設定と画像の整形
            data_sum = np.sum(data, axis=1)
            data[data == 0] = np.nan
            data_avg = np.nanmean(data, axis=1)
            if i == 0:
                datas_avg = np.zeros((data_avg.shape[0], len(folders)))
                datas_sum = np.zeros_like(datas_avg)
            datas_sum[:, i], datas_avg[:, i] = data_sum, data_avg
        print(datas_avg.shape)
        print(datas_avg.dtype)
        np.savetxt(day + '_frond_all_sum.csv', datas_sum, delimiter=',')
        np.savetxt(day + '_frond_all_avg.csv', datas_avg, delimiter=',')
