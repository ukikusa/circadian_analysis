# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys


def npy2csv(a):
    a=0
    return 0


if __name__=='__main__':
    os.chdir('/media/kenya/HD-PLFU3/kenya/171013_jikan/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    # 処理したいデータのフォルダ
    # day = ('./170829-LL2LL-ito-MVX')
    days = ['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX']
    # 解析データのフォルダ
    for day in days:
        frond_folder = day + '/frond'
        for i in sorted(glob.glob(frond_folder + '/*')):
            print(i)
            text = "small_phase_mesh1_avg3"
            img = np.load(os.path.join(i, text + ".npy"))
            x = np.repeat(np.arange(img.shape[1]), img.shape[2])
            y = np.array(list(np.arange(img.shape[2]))*img.shape[1])
            img = img.reshape((img.shape[0], img.shape[1]*img.shape[2]))
            out = np.empty((img.shape[0]+2, img.shape[1]))
            out[0], out[1], out[2:] = x, y, img
            frond_pixel = np.any(out[2:]!=-1,axis=0)
            np.savetxt( os.path.join(i, text + '.csv'),out[:,frond_pixel].transpose())
