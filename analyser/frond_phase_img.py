# utf-8

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt


sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import image_analysis as im
import peak_analysis as pa
from image_analysis import make_colors


def make_fig(data, path):
    plt.rcParams["font.size"] = 5
    plt.rcParams["font.family"] = "sans-serif"
    if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path))
    data.plot(colormap="BuGn", fontsize=5)
    plt.plot(data.index, data.sum(axis=1), color="k")
    plt.xlim(left=0, right=192)  # x軸
    plt.xticks(np.arange(0, 192, 24))  # メモリ
    plt.grid(which="major", axis="x", color="k", linestyle="dotted", lw=0.5)  # 縦の補助線
    plt.savefig(path, format="pdf")
    plt.close()


def frond2fig(data, all_data, path, avg, dT, ymax, offset=0, loc=False, color=False):
    # dataの取り込み,タブ区切りなら拡張子をtsv．　delimiter='\t'でできる気がする試してない．
    data_v = np.array(data.values)
    if ymax > 10000:
        data_v = data_v / np.power(10, int(np.log10(ymax)) - 4)
        ymax = ymax / np.power(10, int(np.log10(ymax)) - 4)
    # 時間の定義
    time = np.arange(data.shape[0], dtype=float) * dT / 60 + offset
    if avg is not False:
        time_move = time[int(avg / 2) : data.shape[0] - int(avg / 2)]
    else:
        time_move = time
    plt.rcParams["font.size"] = 11
    if not os.path.exists(os.path.dirname(path)) and os.path.dirname(path) != "":
        os.makedirs(os.path.dirname(path))
    fig = plt.figure(figsize=(6, 4), dpi=100)  # A4余裕あり．かつ半分
    ax1 = fig.add_subplot(111)
    for i in range(data.shape[1]):
        # pdf一ページに対して，配置するグラフの数．配置するグラフの場所を指定．
        if avg is not False:
            data_move = pa.moving_avg(data_v[:, i], avg=avg)
        else:
            data_move = data_v[:, i]
        # x軸の調整
        ax1.plot(
            time_move,
            data_move,
            linewidth=1,
            color=cm.brg(1 - i / data_v.shape[1]),
            label=data.columns[i],
        )
    ax1.set_ylim([0, ymax])
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=1,
        framealpha=1,
        markerscale=False,
        fontsize=7,
        edgecolor="w",
    )
    plt.vlines(
        np.arange(0, time[-1], 24),
        0,
        ymax,
        colors="k",
        linestyle="dotted",
        label="",
        lw=1,
    )
    # if all_data is not False:
    #    ax2 = ax1.twinx()
    #    ax2.plot(data.index, all_data, color="k")
    #    ax2.set_ylim(bottom=0)
    if all_data is not False:
        # ax2 = ax1.twinx()
        ax1.plot(data.index, all_data, color="k")
        ax1.set_ylim(bottom=0)
    plt.tick_params(labelbottom="off", labelleft="off")
    plt.tick_params(right="off", top="off")
    plt.xlim([0, 192])
    plt.xticks(np.arange(0, 192, 24))

    if loc == "lower right":
        plt.text(time[-1] - int(time[-1] / 2), int(ymax / 10), data.columns[i])
    elif loc == "upper left":
        plt.text(int(time[-1] / 15), ymax - int(ymax / 6), data.columns[i])
    else:
        ax1.get_legend().remove()
        # レイアウト崩れを自動で直してもらう
    # plt.tight_layout()
    # 保存
    plt.savefig(os.path.join(path + ".pdf"), bbox_inches="tight")
    plt.close()
    return 0


def make_frond_phase_imgs(imgs, label_imgs, avg, dT=60, p_range=6, f_range=4, save=""):
    label = np.unique(label_imgs[np.nonzero(label_imgs)])
    data_sqrt = np.empty((imgs.shape[0], label.shape[0]), dtype=np.float64)
    data_sum = np.empty((imgs.shape[0], label.shape[0]), dtype=np.float64)
    theta_imgs = np.full_like(imgs, -1, dtype=np.float64)
    frond_idx = []
    for i in label:
        frond_idx.append("label-" + str(i).zfill(3))
    for i in range(label_imgs.shape[0]):
        img = imgs[i]
        label_img = label_imgs[i]
        for j, k in enumerate(label):
            data_sqrt[i, j] = np.sum(label_img == k)
            data_sum[i, j] = np.sum(img[label_img == k])
    data_avg = data_sum / data_sqrt

    time = np.arange(imgs.shape[0], dtype=float) * dT / 60
    time_move = time[avg // 2 : imgs.shape[0] - avg // 2]
    _peak_t, _peak_v, d_theta, _r2, _peak_point, _func, _d_tau = pa.phase_analysis(
        data_avg, dt_m=dT, p_range=p_range, f_avg=avg, f_range=f_range, offset=0
    )

    for i in range(d_theta.shape[0]):
        for j, k in enumerate(label):
            theta_imgs[i, label_imgs[i] == k] = d_theta[i, j]

    if save is not False:
        avg_df = pd.DataFrame(data_avg, index=time, columns=frond_idx)
        frond2fig(
            avg_df,
            data_sum.sum(axis=1) / data_sqrt.sum(axis=1),
            save + "avg.pdf",
            avg=False,
            dT=dT,
            ymax=5000,
        )
        avg_df.to_csv(save + "avg.csv")

        sum_df = pd.DataFrame(data_sum, index=time, columns=frond_idx)
        frond2fig(
            sum_df,
            data_sum.sum(axis=1),
            save + "sum.pdf",
            avg=False,
            dT=dT,
            ymax=100000,
        )
        sum_df.to_csv(save + "sum.csv")

        sqrt_df = pd.DataFrame(data_sqrt, index=time, columns=frond_idx)
        frond2fig(
            sqrt_df,
            data_sqrt.sum(axis=1),
            save + "sqrt.pdf",
            avg=False,
            dT=dT,
            ymax=6000,
        )
        sqrt_df.to_csv(save + "sqet.csv")

        theta_data = pd.DataFrame(d_theta, index=time, columns=frond_idx)
        theta_data.to_csv(save + "theta.csv")

    return theta_imgs


def img_to_frond_phase(
    folder, label_folder, dark=0, avg=3, dT=60, p_range=12, f_range=5, save=""
):
    # 上の全部まとめたった！！
    data = im.read_imgs(folder)
    data = data.astype(float) - dark
    label_img = im.read_imgs(label_folder)
    # 解析をする．
    imgs_phase = make_frond_phase_imgs(
        imgs=data,
        label_imgs=label_img,
        avg=avg,
        dT=dT,
        p_range=p_range,
        f_range=f_range,
        save=save,
    )
    print(imgs_phase[0, 254, 254])
    imgs_phase[(label_img != 0) * (np.isnan(imgs_phase))] = -2
    print(imgs_phase[0, 254, 254])
    color_phase = make_colors(imgs_phase, grey=-2, black=-1)
    print(color_phase.shape)
    if save is not False:
        # color 画像の保存
        im.save_imgs(save + "frond_phase_color", color_phase, extension="png")
    return color_phase, imgs_phase


# dataはdata,avgで移動平均．前後p_range分より高い点を暫定ピークに．その時の移動平均がpeak_avg．fit_rangeでフィッティング

if __name__ == "__main__":
    os.chdir("/hdd1/Users/kenya/Labo/keisan/python/00data")
    data_file = os.path.join("edit_raw", "lum_min_img")
    label_file = os.path.join("edit_raw", "label_img")

    day = os.path.join(".", "170215-LL2LL-MVX")
    dark = 1578.3
    dT = 60 + 10 / 60
    # 解析データのフォルダ
    data_folder = os.path.join(day, data_file)
    label_folder = os.path.join(day, label_file)
    # 出力先フォルダ
    color, imgs_phase = img_to_frond_phase(
        data_folder,
        label_folder,
        dark=dark,
        avg=3,
        dT=60,
        p_range=12,
        f_range=5,
        save=os.path.join(day, "resut", ""),
    )

    day = os.path.join(".", "170613-LD2LL-ito-MVX")
    dT = 60
    dark = 1565.8
    # # 解析データのフォルダ
    data_folder = os.path.join(day, data_file)
    label_folder = os.path.join(day, label_file)
    # 出力先フォルダ
    color, imgs_phase = img_to_frond_phase(
        data_folder,
        label_folder,
        dark=dark,
        avg=3,
        dT=60,
        p_range=12,
        f_range=5,
        save=os.path.join(day, "resut", ""),
    )

    day = os.path.join(".", "170829-LL2LL-ito-MVX")
    dT = 60
    dark = 1563.0
    # 解析データのフォルダ
    data_folder = os.path.join(day, data_file)
    label_folder = os.path.join(day, label_file)
    # 出力先フォルダ
    color, imgs_phase = img_to_frond_phase(
        data_folder,
        label_folder,
        dark=dark,
        avg=3,
        dT=60,
        p_range=12,
        f_range=5,
        save=os.path.join(day, "resut", ""),
    )
