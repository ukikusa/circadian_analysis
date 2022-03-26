# -*- coding: utf-8 -*-
"""phase diff - distance."""

import os

# import glob
# import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# 下２行は，リモートで操作している場合
import matplotlib as mpl

mpl.use("Agg")


def phase_diff(phase_img, pp=False, scale=0.033, title="", x_max=3.5, x_range=0.05):
    """Phase diff - distance from image."""
    x, y = np.where(~np.isnan(phase_img) * phase_img >= 0)  # 位相が定義されている座標
    xy = np.array((x, y)).T
    phase_d = phase_img[x, y]  # 上に対応する位相のリスト
    size = np.size(phase_d)  # 使うデータの個数
    idx = range(np.size(phase_d))
    tri_dx = np.triu_indices(size, 1)  # 上三角行列のIndex
    xx, yy = np.meshgrid(idx, idx)
    dis = np.linalg.norm(xy[xx[tri_dx]] - xy[yy[tri_dx]], axis=1)  # 距離
    dis = dis * scale
    diff = np.abs(phase_d[xx[tri_dx]] - phase_d[yy[tri_dx]])  # 位相差
    diff[diff > 0.5] = 1 - diff[diff > 0.5]
    if not os.path.exists(os.path.dirname(pp)):
        os.makedirs(os.path.dirname(pp))
    if pp is not False:  # pdfの保存
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
        sc_bottom = sc_left = 0.1
        sc_width = 0.85
        sc_height = 0.65
        space = 0.01
        hst_height = 0.15
        ####################################################
        # ボックスプロットをする
        ax = plt.axes([sc_bottom, sc_left, sc_width, sc_height])
        x_avg = np.arange(0, x_max, x_range)  # 村中博論に合わせて0.05mm感覚で
        diff_box = []
        for c, i in enumerate(x_avg):
            idx = (dis < (i + x_range)) * (dis >= i)
            diff_box.append(diff[idx])  # ボックスプロットは要素をタプルで欲しがる
        whiskerprops = {"linestyle": "solid", "linewidth": -1, "color": "k"}
        ax.boxplot(
            diff_box,
            positions=x_avg + x_range * 0.5,
            showmeans=True,
            meanline=True,
            meanprops=dict(color="g", linewidth=2),
            medianprops=dict(color="r", linewidth=2),
            whis="range",
            widths=x_range,
            whiskerprops=whiskerprops,
            showcaps=False,
        )
        ax.set_ylim(0, 0.5)

        ax.set_xlim(0, x_max)
        ax.set_xlabel(r"cell-to-cell distance ($\mu m $)")
        ax.set_ylabel(r"phase diffrence ($rad / \pi $)")
        ax.set_xticks(np.arange(0, x_max, 0.5))  # x軸の調整．トリッキーなので…
        ax.set_xticklabels(np.arange(0, x_max, 0.5))  # 同上

        ########################################################
        # ヒストグラムを上につける
        ax_x = plt.axes([sc_left, sc_bottom + sc_height + space, sc_width, hst_height])
        ax_x.set_xlim(0, x_max)
        ax_x.set_xticklabels([])
        ax_x.hist(dis, bins=int(x_max / x_range), histtype="stepfilled")
        ax_x.set_ylim(bottom=0)

        ax_x.set_title(title)
        plt.savefig(pp)
        fig.clf()
    return diff, dis


def phase_diff_all(
    folder,
    label,
    curve,
    point,
    phase_path="theta.tif",
    save_path="phase_diff.svg",
    scale=1,
    dbg=False,
):
    """Loop phase_diff_all."""
    # scale 33um らしい．僕の修論によると．
    data_folder = os.path.join(folder, label, curve)
    if not os.path.exists(os.path.join(data_folder, "theta.npy")):
        return 0
    theta = np.load(os.path.join(data_folder, "theta.npy"))
    theta = theta[:, 60:200, 20:160]
    phase = theta[point]
    if dbg is True:
        phase[~np.isnan(phase)] = np.random.rand(np.sum(~np.isnan(phase)))
    plt.rcParams["font.size"] = 9
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["xtick.major.width"] = 2.0  # x軸主目盛り線の線幅
    plt.rcParams["ytick.major.width"] = 2.0  # y軸主目盛り線の線幅
    plt.rcParams["axes.linewidth"] = 2.0
    pp = os.path.join(data_folder, save_path)

    if np.sum(~np.isnan(phase) * phase >= 0) > 100:
        phase_diff(phase, pp=pp, scale=scale, title=str(point) + "h")
    plt.close()
    return 0


if __name__ == "__main__":
    # os.getcwd() # これでカレントディレクトリを調べられる

    DAY = ["170215-LL2LL-MVX", "170613-LD2LL-ito-MVX", "170829-LL2LL-ito-MVX"]
    CURVE = "tau_mesh-1_avg-3_prange-7_frange-5"
    for day_i in DAY:
        folder_i = os.path.join(
            "/hdd1/Users/kenya/Labo/keisan/python/00data", day_i, "frond_190920"
        )
        frond_idx = pd.read_csv(
            os.path.join(
                "/hdd1/Users/kenya/Labo/keisan/python/00data",
                day_i,
                "frond_number_edit.csv",
            ),
            index_col=0,
        )

        for label_i in sorted(os.listdir(folder_i)):
            print(label_i)
            point = frond_idx["pix_usable_idx"][label_i] + 60
            if frond_idx["use_e_idx"][label_i] > point:
                phase_diff_all(
                    folder_i,
                    label_i,
                    CURVE,
                    save_path=os.path.join("phase_diff", str(point) + "h.svg"),
                    point=point,
                    scale=0.033,
                )
