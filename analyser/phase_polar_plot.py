# -*- coding: utf-8 -*-
"""位相とRを円上にプロットする"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from phase2R import phase2Rs


def phase_plot(theta, r=False):
    plt.rcParams["axes.linewidth"] = 0  # x軸目盛りの太さ
    plt.rcParams["xtick.bottom"] = "off"
    plt.rcParams["ytick.left"] = "off"
    plt.rcParams["ytick.labelsize"] = 0
    if r is False:
        r = np.ones_like(np.array(theta.values))
    plt.figure(figsize=(6, 6), dpi=100)  # A4余裕あり．かつ半分
    plt.polar(
        theta, r, color="w", lw=0, marker=".", markersize=10, mec="r", mfc="r"
    )  # 極座標グラフのプロット
    plt.ylim(0, 1.02)
    plt.yticks([0, 1])
    plt.grid(which="major", color="black", linestyle="-")
    plt.xticks(
        np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2, 0]),
        [r"$0$", r"$1/2\pi$", r"$\pi$", r"$3/2\pi$"],
    )
    plt.show()


def phase3_plot(theta, r=False):
    plt.rcParams["axes.linewidth"] = 0  # x軸目盛りの太さ
    plt.rcParams["xtick.bottom"] = "off"
    plt.rcParams["ytick.left"] = "off"
    plt.rcParams["ytick.labelsize"] = 0
    if r is False:
        r = np.ones_like(np.array(theta.values))
    plt.figure(figsize=(7, 5), dpi=100)  # A4余裕あり．かつ半分
    for (i, s) in enumerate(theta.index):
        plt.subplot(2, 3, i + 1, projection="polar")
        plt.polar(
            theta.values[i, :],
            r[i, :],
            color="w",
            lw=0,
            marker=".",
            markersize=10,
            mec="r",
            mfc="r",
            label=s,
        )  # 極座標グラフのプロット
        plt.ylim(0, 1.1)
        plt.yticks([0, 1])
        plt.xlabel(s)
        plt.grid(which="major", color="black", linestyle="-")
        plt.xticks(
            np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2, 0]),
            [r"$0$", r"$1/2\pi$", r"$\pi$", r"$3/2\pi$"],
        )
    plt.tight_layout()
    plt.savefig("tmp.pdf")


def r3_plot(
    theta, theta_all, blue=False, r=False, r_all=False, pdf_save=False, hozyo=False
):
    """フロンド毎の位相と位相と全体の位相を投げると極座標プロットする．削除できそうな関数
    3つまでしかプロットできない
    
    Args:
        theta ([type]): 位相データの集合
        theta_all ([type]): 全体の位相の集合
        blue([list]): indexのリスト．白抜きにする
        r (bool, optional): 個々の位相のr. Defaults to False.
        r_all (bool, optional): 全体で見たときのr. Defaults to False.
        pdf_save (bool, optional): 保存先. Defaults to False.
    """
    plt.rcParams["axes.linewidth"] = 0  # x軸目盛りの太さ
    plt.rcParams["xtick.bottom"] = "off"
    plt.rcParams["ytick.left"] = "off"
    plt.rcParams["ytick.labelsize"] = 0
    if r is False:
        r = np.ones_like(np.array(theta.values))
    plt.figure(figsize=(7, 5), dpi=100)  # A4余裕あり．かつ半分
    for (i, s) in enumerate(theta.index):
        plt.subplot(1, 3, i + 1, projection="polar")
        # 外周．実践にしたいので円で書く．
        plt.polar(np.linspace(0, 2 * np.pi, 100), [1] * 100, lw=1, color="k", marker="")
        # plot　黒で
        plt.polar(
            theta.values[i, :],
            r[i, :],
            color="w",
            lw=0,
            marker=".",
            markersize=10,
            mec="k",
            mfc="k",
            label=s,
        )  # 極座標グラフのプロット
        if blue is not False:  # 別の色でプロットしたい場合
            r_b = np.ones_like(np.array(blue.values))
            plt.polar(
                blue.values[i, :],
                r_b[i, :],
                color="w",
                lw=0,
                marker="o",
                markersize=5,
                mec="k",
                mfc="w",
                label=s,
            )  # 極座標グラフのプロット
        plt.polar(
            theta_all[i],
            r_all[i],
            lw=0,
            marker=".",
            markersize=10,
            mec="r",
            mfc="r",
            label=s,
        )  # 全体のプロット
        plt.ylim(0, 1.1)  # 1までにすると，1のところにプロットできない
        if hozyo is not False:  # 有意差を示す補助線．点線にするのはGlidの方がやりやすい．
            plt.yticks([hozyo[i]])

        plt.xlabel(str(s) + "h; R=" + "{:.2f}".format(r_all[i]))
        plt.grid(which="major", color="black", linestyle=":")
        plt.xticks(
            np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2, 0]),
            [r"$0$", r"$1/2\pi$", r"$\pi$", r"$3/2\pi$"],
        )
    plt.tight_layout()
    if pdf_save is not False:
        if not os.path.exists(os.path.basename(pdf_save)):
            os.makedirs(os.path.basename(pdf_save))
        plt.savefig(pdf_save)


# 極方程式
if __name__ == "__main__":
    os.chdir(
        os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data")
    )
    data_file = "avg-3_prange-7_frange-5_phase.csv"

    days = "170829-LL2LL-ito-MVX"

    data_folder = os.path.join(days, "frond_190920_individual")
    pdf_save = os.path.join(data_folder, "frond_r.pdf")
    dataframe = pd.read_csv(
        os.path.join(data_folder, data_file), dtype=np.float64, index_col=0
    )
    dataframe = dataframe.replace(-1, np.nan)
    theta = dataframe.loc[[48, 96, 144], :] * 2 * np.pi

    r_all, n, e = phase2Rs(dataframe.loc[[48, 96, 144], :].values)
    e, n = np.array(e), np.array(n)
    theta_all = np.angle(e / n)
    print(theta_all, r_all)
    r3_plot(theta, theta_all=theta_all, r=False, r_all=r_all, pdf_save=pdf_save)
    #####################################################
    days = "170613-LD2LL-ito-MVX"

    data_folder = os.path.join(days, "frond_190920_individual")
    pdf_save = os.path.join(data_folder, "frond_r.pdf")

    dataframe = pd.read_csv(
        os.path.join(data_folder, data_file), dtype=np.float64, index_col=0
    )
    dataframe = dataframe.replace(-1, np.nan)
    theta = dataframe.loc[[48, 96, 144], :] * 2 * np.pi

    r_all, n, e = phase2Rs(dataframe.loc[[48, 96, 144], :].values)
    e, n = np.array(e), np.array(n)
    theta_all = np.angle(e / n)
    print(theta_all, r_all)
    r3_plot(theta, theta_all=theta_all, r=False, r_all=r_all, pdf_save=pdf_save)

    #####################################################
    days = "170215-LL2LL-MVX"

    data_folder = os.path.join(days, "frond_190920_individual")
    pdf_save = os.path.join(data_folder, "frond_r.pdf")

    dataframe = pd.read_csv(
        os.path.join(data_folder, data_file), dtype=np.float64, index_col=0
    )
    dataframe = dataframe.replace(-1, np.nan)
    theta = dataframe.iloc[[48, 96, 144], :] * 2 * np.pi

    r_all, n, e = phase2Rs(dataframe.iloc[[48, 96, 144], :].values)
    e, n = np.array(e), np.array(n)
    theta_all = np.angle(e / n)
    print(theta_all, r_all)
    r3_plot(theta, theta_all=theta_all, r=False, r_all=r_all, pdf_save=pdf_save)

    #####################################################
