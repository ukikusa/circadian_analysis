# -*-coding: utf-8 -*-
"""area."""

import os

import matplotlib as mpl
# mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import numpy as np


def move_avg(x, data_v, avg):
    data_t, data_n = data_v.shape
    x_m = x[int(avg * 0.5):data_t - int(avg * 0.5)]
    data_v_m = np.zeros([data_t - avg + 1, data_n], dtype=np.float64)
    for i in range(avg):
        data_v_m = data_v_m + data_v[i:data_t - avg + i + 1]
    data_v_m = data_v_m / avg
    return x_m, data_v_m


def per_dif(x, data_v, dif_n):
    x = x[:-dif_n]
    data_dif = data_v[dif_n:] - data_v[:-dif_n]
    per_dif = data_dif / data_v[:-dif_n]
    return x, per_dif


def plot_dif(x, y, save_path, title=False, ylabel='increace_rate(%)', log=False, label=False, use_day=8):
    fig = plt.figure(figsize=(6, 4), dpi=100)
    time, data_n = y.shape
    for i in range(data_n):
        plt.plot(x, y[:, i], lw=0.5, color=cm.jet(i / data_n))
    if log is True:
        plt.yscale("log")
        # plt.ylim(bottom=0.01, top=100)
    plt.xlim(0, 24 * use_day)
    plt.xticks(range(0, 24 * use_day, 24))
    plt.xlabel("time(h)")
    plt.ylabel(ylabel)
    plt.grid(which='major', color='black', linestyle=':')
    if ylabel == 'increace_rate(%)' and log is False:
        plt.ylim(bottom=-0.5)
    if label is not False:
        plt.legend(label)
    if title is not False:
        plt.title(title, fontsize=6)
    plt.savefig(save_path, format='pdf')
    plt.close()


def plot_N_dif(x, y, save_path, title=False, log=False, label=False):
    fig = plt.figure(figsize=(6, 4), dpi=100)
    time, data_n = y.shape
    for i in range(data_n):
        plt.plot(x[:, i], y[:, i], lw=0, color=cm.jet(i / data_n), marker=".")
    if log is True:
        plt.yscale("log")
        # plt.ylim(bottom=0.01, top=100)
    plt.xlim(left=0)
    plt.xlabel("N")
    plt.ylabel("increace_rate(%)")
    plt.grid(which='major', color='black', linestyle=':')
    if log is False:
        plt.ylim(bottom=-0.5)
    if label is not False:
        plt.legend(label)
    if title is not False:
        plt.title(title, fontsize=6)
    plt.savefig(save_path, format='pdf')
    plt.close()


def align_nan(data):
    data_new = np.empty_like(data)
    data_new[:] = np.nan
    for i in range(data_new.shape[1]):
        data_i = data[:, i]
        data_new[:data_i.shape[0] - np.sum(np.isnan(data_i)), i] = data_i[~np.isnan(data_i)]
    return data_new


def area_analysis(data, save, dt=60, avg=False, label=False, use_day=8):
    data_v = data.values
    data_v[data_v == 0] = np.nan
    x = list(data.index)
    if avg is not False:
        x, data_v = move_avg(x, data_v, avg)
    dif_n = 24 * int(60 / dt)
    x24, per_dif24 = per_dif(x, data_v, dif_n=dif_n)
    x_dif = x[:-dif_n]
    data_dif = data_v[dif_n:] - data_v[:-dif_n]
    folder = os.path.dirname(save)
    if os.path.exists(folder) is False:
        os.makedirs(folder)
    pp = PdfPages(save)
    plot_dif(x, data_v, save_path=pp, title='averate_' + str(avg) + '-diff_24', ylabel='area(pixel)', label=label, use_day=use_day)
    plot_dif(x, data_v, save_path=pp, title='averate_' + str(avg) + '-diff_24', ylabel='area(pixel)', label=label, log=True, use_day=use_day)
    plot_dif(x_dif, data_dif, save_path=pp, title='average_-diff_24', ylabel='diff(pixel)', label=label, use_day=use_day)
    plot_N_dif(data_v[:-dif_n], per_dif24, save_path=pp, title='average_' + str(avg) + '-diff_24', label=label)
    plot_N_dif(data_v[:-dif_n], per_dif24, save_path=pp, title='average_' + str(avg) + '-diff_24', log=True, label=label)
    ###############################
    # 揃えた方
    ###########################
    data_v_align = align_nan(data_v)
    plot_dif(x, data_v_align, save_path=pp, title='align_average_' + str(avg), ylabel='area(pixel)', label=label, use_day=use_day)
    pp.close()
