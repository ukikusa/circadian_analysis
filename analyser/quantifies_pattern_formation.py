# -*- coding: utf-8 -*-
"""
    This function quantifies spatio-temporal pattern formation in imaging data..

https://academic.oup.com/bioinformatics/article/33/19/3072/3859179
"""

import os

import numpy as np
from scipy.spatial.distance import cdist


def phase_diff(x, y):
    return np.arctan2(np.sin(x - y), np.cos(x - y))


def Morans_i(data, decay=-2):
    coordinate = np.where(~np.isnan(data) * data >= 0)
    phase_data = data[coordinate] * 2 * np.pi
    coordinate_np = np.array(coordinate).T
    weight = cdist(coordinate_np, coordinate_np)
    weight[weight != 0] = weight[weight != 0] ** decay
    total_mean = np.arctan2(np.sum(np.sin(phase_data)), np.sum(np.cos(phase_data)))
    data_diff = phase_diff(phase_data, total_mean)  # meanとの引き算を先にするb
    MI = (
        np.sum(weight * np.outer(data_diff, data_diff))
        / np.average(np.square(data_diff))
        / np.sum(weight)
    )
    return MI


if __name__ == "__main__":

    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python"))
    # 処理したいデータのフォルダ

    days = ["170215-LL2LL-MVX", "170613-LD2LL-ito-MVX", "170829-LL2LL-ito-MVX"]
    file_ = os.path.join(
        "result", "170613-LD2LL-ito-MVX", "label-002_n214", "theta.npy"
    )
    data = np.load(file_)
    # data[np.where(~np.isnan(data) * data >= 0)] = 1
    # data[np.where(~np.isnan(data) * data >= 0)] = np.random.rand(
    #     data[np.where(~np.isnan(data) * data >= 0)].size
    # )
    for i in range(data.shape[0]):
        Morans_i(data[i])
