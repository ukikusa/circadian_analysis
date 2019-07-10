# -*- coding: utf-8 -*-
"""
    Quantifies spatio-temporal pattern formation in imaging data.

https://academic.oup.com/bioinformatics/article/33/19/3072/3859179
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import numpy as np


def phase_diff(x, y):
    return np.arctan2(np.sin(x - y), np.cos(x - y))


def Morans_i(data, decay=-2):
    coordinate = np.where(~np.isnan(data) * data >= 0)
    data[coordinate] = data[coordinate] * 2 * np.pi
    coordinate_np = np.array(coordinate).T
    tmp_index = np.arange(len(coordinate[0]))
    ii, jj = np.meshgrid(tmp_index, tmp_index)  # 全ての組み合わせのリストを作成
    ii_coor, jj_coor = np.reshape(coordinate_np[ii].T, (2, -1)), np.reshape(coordinate_np[jj].T, (2, -1))
    weight = np.linalg.norm(ii_coor - jj_coor, axis=0) ** decay
    weight[np.isinf(weight)] = 0
    total_mean = np.arctan2(np.sum(np.sin(data[coordinate])), np.sum(np.cos(data[coordinate])))
    data_diff = phase_diff(data, total_mean)
    data_i = data_diff[tuple(ii_coor)]
    data_j = data_diff[tuple(jj_coor)]
    MI = np.dot(weight, data_i * data_j) / np.sum(np.square(data_diff[coordinate])) / np.sum(weight) * data[coordinate].size
    print(MI)
    return MI

if __name__ == "__main__":

    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python"))
    # 処理したいデータのフォルダ

    days = (['170215-LL2LL-MVX', '170613-LD2LL-ito-MVX', '170829-LL2LL-ito-MVX'])
    file_ = os.path.join("result", "170613-LD2LL-ito-MVX", "label-002_n214", "theta.npy")
    data = np.load(file_)
    data[np.where(~np.isnan(data) * data >= 0)] = 1
    data[np.where(~np.isnan(data) * data >= 0)] = np.random.rand(data[np.where(~np.isnan(data) * data >= 0)].size)
    for i in range(data.shape[0]):
        Morans_i(data[i])
