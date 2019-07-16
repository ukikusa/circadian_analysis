# -*- coding: utf-8 -*-
"""
    This function quantifies spatio-temporal pattern formation in imaging data..

https://academic.oup.com/bioinformatics/article/33/19/3072/3859179
"""

import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def morans_i(img, decay=-2):
    """Function to caliculate Moran’s I from phase image.

    Args:
        img (np.float): A numpy array with phases defined from 0 to 1.
        decay (int, optional): The decay parameter specifying the typical interaction range.
. Defaults to -2.

    Returns:
         Moran’s I : float
    """
    coordinate = np.where(~np.isnan(img) * img >= 0)
    phase_data = img[coordinate] * 2 * np.pi
    coordinate_np = np.array(coordinate).T
    weight = cdist(coordinate_np, coordinate_np)  # 距離を求める
    weight[weight != 0] = weight[weight != 0] ** decay
    phase_mean = np.arctan2(np.sum(np.sin(phase_data)), np.sum(np.cos(phase_data)))
    diff_phase = np.arctan2(
        np.sin(phase_data - phase_mean), np.cos(phase_data - phase_mean)
    )  # meanとの引き算を先にする
    m_i = (
        np.sum(weight * np.outer(diff_phase, diff_phase))
        / np.average(np.square(diff_phase))
        / np.sum(weight)
    )
    return m_i


def mi_time_series(imgs, mi_df=pd.DataFrame(), colome="MI"):
    """Function to caliculate Moran’s I from phase images.

    Args:
        imgs (np.array): A numpy array with phases defined from 0 to 1. Axis=0: time series
        mi_df ([type], optional): return. Defaults to pd.DataFrame().
        colome (str, optional): colome name. Defaults to "MI".

    Returns:
        data frame: mi dataframe
    """
    mi_np = np.empty((imgs.shape[0]), dtype=np.float64)
    for i in range(imgs.shape[0]):
        mi_np[i] = morans_i(imgs[i])
    mi_df[colome] = mi_np
    return mi_df


def mi_fronds(file_list):
    """Test"""
    mi_df = pd.DataFrame()
    for i in file_list:
        imgs = np.load(file_list[i])
        mi_df = mi_time_series(
            imgs, mi_df, os.path.splitext(os.path.basename(file_list[i]))[0]
        )
    return mi_df


if __name__ == "__main__":

    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python"))
    # 処理したいデータのフォルダ

    DAYS = ["170215-LL2LL-MVX", "170613-LD2LL-ito-MVX", "170829-LL2LL-ito-MVX"]
    FILE = os.path.join("result", DAYS[1], "label-002_n214", "theta.npy")
    DATA = np.load(FILE)
    # data[np.where(~np.isnan(data) * data >= 0)] = 1
    # data[np.where(~np.isnan(data) * data >= 0)] = np.random.rand(
    #     data[np.where(~np.isnan(data) * data >= 0)].size
    # )
    MI_DF = mi_time_series(DATA, mi_df=pd.DataFrame(), colome="MI")
