import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from phase2R import phase2R
from phase2R import phase2Rs


def phase_plot(theta, r=False):
    plt.rcParams['axes.linewidth'] = 0  # x軸目盛りの太さ
    plt.rcParams['xtick.bottom'] = 'off'
    plt.rcParams['ytick.left'] = 'off'
    plt.rcParams['ytick.labelsize'] = 0
    if r is False:
        r = np.ones_like(np.array(theta.values))
    plt.figure(figsize=(6, 6), dpi=100)  # A4余裕あり．かつ半分
    plt.polar(theta, r, color='w', lw=0, marker='.', markersize=10, mec='r', mfc='r')  # 極座標グラフのプロット
    plt.ylim(0, 1.02)
    plt.yticks([0, 1])
    plt.grid(which='major', color='black', linestyle='-')
    plt.xticks(np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2, 0]), ['$0$', '$1/2\pi$', '$\pi$', '$3/2\pi$'])
    plt.show()


def phase3_plot(theta, r=False):
    plt.rcParams['axes.linewidth'] = 0  # x軸目盛りの太さ
    plt.rcParams['xtick.bottom'] = 'off'
    plt.rcParams['ytick.left'] = 'off'
    plt.rcParams['ytick.labelsize'] = 0
    if r is False:
        r = np.ones_like(np.array(theta.values))
    plt.figure(figsize=(7, 5), dpi=100)  # A4余裕あり．かつ半分
    for (i, s) in enumerate(theta.index):
        plt.subplot(2, 3, i + 1, projection="polar")
        plt.polar(theta.values[i, :], r[i, :], color='w', lw=0, marker='.', markersize=10, mec='r', mfc='r', label=s)  # 極座標グラフのプロット
        plt.ylim(0, 1.1)
        plt.yticks([0, 1])
        plt.xlabel(s)
        plt.grid(which='major', color='black', linestyle='-')
        plt.xticks(np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2, 0]), ['$0$', '$1/2\pi$', '$\pi$', '$3/2\pi$'])
    plt.tight_layout()
    plt.savefig('tmp.pdf')


def r3_plot(theta, theta_all, r=False, r_all=False, pdf_save=False):
    plt.rcParams['axes.linewidth'] = 0  # x軸目盛りの太さ
    plt.rcParams['xtick.bottom'] = 'off'
    plt.rcParams['ytick.left'] = 'off'
    plt.rcParams['ytick.labelsize'] = 0
    if r is False:
        r = np.ones_like(np.array(theta.values))
    plt.figure(figsize=(7, 5), dpi=100)  # A4余裕あり．かつ半分
    for (i, s) in enumerate(theta.index):
        plt.subplot(2, 3, i + 1, projection="polar")
        plt.polar(theta.values[i, :], r[i, :], color='w', lw=0, marker='.', markersize=10, mec='b', mfc='b', label=s)  # 極座標グラフのプロット
        plt.polar(theta_all[i], r_all[i], color='w', lw=0, marker='.', markersize=10, mec='r', mfc='r', label=s)  # 極座標グラフのプロット
        plt.ylim(0, 1.1)
        plt.yticks([0, 1])
        plt.xlabel(s)
        plt.grid(which='major', color='black', linestyle='-')
        plt.xticks(np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2, 0]), ['$0$', '$1/2\pi$', '$\pi$', '$3/2\pi$'])
    plt.tight_layout()
    if pdf_save is not False:
        plt.savefig(pdf_save)


# 極方程式
if __name__ == '__main__':
    # os.chdir(os.path.join("/Users", "kenya", "keisan", "python", "00data"))
    os.chdir(os.path.join("/hdd1", "kenya", 'Labo', "keisan", "python", "00data"))
    data_file = '2018-01-14-frond_photo_avgphase.csv'

    days = "./170829-LL2LL-ito-MVX"

    data_folder = os.path.join(days, 'result', 'pdf')
    pdf_save = os.path.join(data_folder, 'frond_r.pdf')
    dataframe = pd.read_csv(os.path.join(data_folder, data_file), dtype=np.float64, index_col=0)
    dataframe = dataframe.replace(-1, np.nan)
    theta = dataframe.loc[[48, 96, 144], :] * 2 * np.pi

    r_all, n, e = phase2Rs(dataframe.loc[[48, 96, 144], :].values)
    e, n = np.array(e), np.array(n)
    theta_all = np.angle(e / n)
    print(theta_all, r_all)
    r3_plot(theta, theta_all=theta_all, r=False, r_all=r_all, pdf_save=pdf_save)

    #####################################################
    days = "./170613-LD2LL-ito-MVX"

    data_folder = os.path.join(days, 'result', 'pdf')
    pdf_save = os.path.join(data_folder, 'frond_r.pdf')

    dataframe = pd.read_csv(os.path.join(data_folder, data_file), dtype=np.float64, index_col=0)
    dataframe = dataframe.replace(-1, np.nan)
    theta = dataframe.loc[[48.5, 96.5, 144.5], :] * 2 * np.pi

    r_all, n, e = phase2Rs(dataframe.loc[[48.5, 96.5, 144.5], :].values)
    e, n = np.array(e), np.array(n)
    theta_all = np.angle(e / n)
    print(theta_all, r_all)
    r3_plot(theta, theta_all=theta_all, r=False, r_all=r_all, pdf_save=pdf_save)

    #####################################################
    days = "./170215-LL2LL-MVX"

    data_folder = os.path.join(days, 'result', 'pdf')
    pdf_save = os.path.join(data_folder, 'frond_r.pdf')

    dataframe = pd.read_csv(os.path.join(data_folder, data_file), dtype=np.float64, index_col=0)
    dataframe = dataframe.replace(-1, np.nan)
    theta = dataframe.iloc[[48, 96, 144], :] * 2 * np.pi

    r_all, n, e = phase2Rs(dataframe.iloc[[48, 96, 144], :].values)
    e, n = np.array(e), np.array(n)
    theta_all = np.angle(e / n)
    print(theta_all, r_all)
    r3_plot(theta, theta_all=theta_all, r=False, r_all=r_all, pdf_save=pdf_save)

    #####################################################
