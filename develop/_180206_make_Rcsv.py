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


def make_R_csv(day):
    frond_folder = sorted(glob.glob(os.path.join(day, 'frond', '*')))
    R_data = np.load(os.path.join(frond_folder[0], 'small_R.npy'))
    R_data_all = np.empty((len(frond_folder), len(R_data)))
    for i, folder in enumerate(frond_folder):
        print(i)
        R_data_all[i] = np.load(os.path.join(folder, 'small_R.npy'))
    R_data_save = pd.DataFrame(R_data_all, index=frond_folder)
    R_data_save.to_csv(os.path.join(day, 'small_R.csv'))
    return 0


def make_period_csv(day):
    frond_folder = sorted(glob.glob(os.path.join(day, 'frond', '*')))
    period_data = np.load(os.path.join(frond_folder[0], 'small_period_mesh1_avg3.npy'))
    period_data[period_data == -1] = np.nan
    period = (np.nanmean(period_data, axis=(1, 2)))
    period_all = np.empty((len(frond_folder), len(period)))
    for i, folder in enumerate(frond_folder):
        if i == 0:
            period_all[i] = period
        else:
            period_data = np.load(os.path.join(frond_folder[i], 'small_period_mesh1_avg3.npy'))
            period_data[period_data < 18] = np.nan
            period_data[period_data > 36] = np.nan
            period_all[i] = np.nanmean(period_data, axis=(1, 2))
    period_save = pd.DataFrame(period_all, index=frond_folder)
    period_save.to_csv(os.path.join(day, 'small_period_avg3.csv'))
    return 0


if __name__ == '__main__':
    # os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    days = ['./170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX', './170215-LL2LL-MVX']
    for day in days:
        # make_R_csv(day)
        make_period_csv(day)
