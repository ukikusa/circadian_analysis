# -*- coding: utf-8 -*-
"""Make figure."""

import os
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


def start_plot(save_path, size_x=11.69, size_y=8.27):
    """ start plot

    plot for malti page

    Args:
        save_path: [description]
        size_x: [description] (default: {11.69})
        size_y: [description] (default: {8.27})

    Returns:
        pp, fig, ax=[]
    """
    # 文字サイズとかの設定
    plt.rcParams["font.size"] = 5
    plt.rcParams["font.family"] = "sans-serif"
    if (
        os.path.exists(os.path.dirname(save_path)) is False
        and os.path.dirname(save_path) != ""
    ):
        os.makedirs(os.path.dirname(save_path))
    print("pdf作成を開始します")
    pp = PdfPages(save_path)
    fig = plt.figure(figsize=(size_x, size_y), dpi=100)
    ax = []
    return pp, fig, ax


def pdf_save(pp):
    plt.tight_layout()  # レイアウト
    plt.savefig(pp, format="pdf")
    plt.clf()


def multi_plot(
    x,
    y,
    save_path,
    peak=False,
    func=False,
    r=False,
    label=False,
    y_min=0,
    y_max=None,
    plt_x=5,
    plt_y=4,
    size_x=11.69,
    size_y=8.27,
):
    """A function that plots multiple graphs on one page."""
    # yはたくさんある前提.
    # peakを中心に前後rでフィティングをしたとする．
    # 28個の時系列折れ線グラフを一つのPDFファイルとして出力する．
    pp, fig, ax = start_plot(save_path, size_x=size_x, size_y=size_y)
    plt_n = plt_x * plt_y  # 一つのグラフへのプロット数

    if label is False:
        label = np.empty(y.shape[1])
        label[:] = False

    ###############
    # pdfの保存関数 ここから
    ##############

    def plot_data(
        fig, ax, x, y, y_min=y_min, plt_x=plt_x, plt_y=plt_y, i=0, label=False
    ):
        # print(ax)
        ax.append(fig.add_subplot(plt_x, plt_y, i + 1))
        # プロット
        ax[i].plot(x, y, linewidth=0, marker=".")
        # 軸の調整とか
        ax[i].set_xlim(left=0)  # x軸
        ax[i].set_ylim(y_min, y_max)  # y軸
        ax[i].set_xticks(np.arange(0, x[-1], 24) - x[0])  # メモリ
        ax[i].grid(
            which="major", axis="x", color="k", linestyle="dotted", lw=0.5
        )  # 縦の補助線
        ax[i].set_title(label)
        ax[i].tick_params(labelbottom=True, labelleft=True, labelsize=5)
        return fig, ax

    def plot_peak_fit(ax, x, peak, func, r, i=0):
        peak = peak[~np.isnan(peak)]
        for count, j in enumerate(peak):
            ax[i].plot(
                x[int(j - r) : int(j + r)],
                np.poly1d(func[count])(x[int(j - r) : int(j + r)]),
                "-r",
                lw=1,
            )
        return ax

    ##########
    # ループ #
    ##########
    for i in range(y.shape[1]):
        # 1pageに対して，配置する graf の数．配置する graf の場所を指定．
        i_mod = i % plt_n
        fig, ax = plot_data(
            fig,
            ax,
            x,
            y=y[:, i],
            y_min=y_min,
            plt_x=plt_x,
            plt_y=plt_y,
            i=i_mod,
            label=label[i],
        )
        # print(i)
        #####################
        # fittingのプロット #
        #####################
        if peak is not False:
            ax = plot_peak_fit(ax, x, peak[:, i], func[:, i], r, i=i_mod)
        #############
        # pdfの保存
        #################
        if np.mod(i, plt_n) == plt_n - 1:
            pdf_save(pp)
            ax = []
    #########
    # 残ったやつのPDFの保存
    ############
    if np.mod(i, plt_n) != plt_n - 1:
        pdf_save(pp)
    plt.clf()
    pp.close()


def make_hst_fig(
    save_file,
    x,
    y,
    min_x=0,
    max_x=None,
    min_y=0,
    max_y=None,
    max_hist_x=None,
    max_hist_y=None,
    bin_hist_x=100,
    bin_hist_y=100,
    xticks=False,
    yticks=False,
    xticklabels=[],
    yticklabels=[],
    pdfpages=False,
    avg=False,
    xlabel="",
    ylabel="",
    box=False,
    per=True,
    title=False,
):
    """Create a graph with histogram.
    You can also find correlation coefficients or plot box plots."""
    # Min_x-max_y:散布図の範囲．max_hist:度数分布の最大値，bin:度数分布の分割.
    x, y = x.astype(np.float64), y.astype(np.float64)
    folder = os.path.dirname(save_file)
    if os.path.exists(folder) is False:
        os.makedirs(folder)
    ###############################
    # graph position
    sc_bottom = sc_left = 0.1
    sc_width = sc_height = 0.65
    space = 0.01
    hst_height = 0.2
    ####################################################
    # if pdfpages is False:
    fig = plt.figure(1, figsize=(6, 4), dpi=100)
    ax = plt.axes([sc_bottom, sc_left, sc_width, sc_height])
    xy = np.vstack([x, y])
    z = scipy.stats.gaussian_kde(xy)(xy)  # 色を決める．
    ax.scatter(x, y, c=z, marker=".", s=1)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    #################################################
    # 相関係数 ヒストグラムをxもyも出したときのみ対応
    #################################################
    if per is True:
        r, p = scipy.stats.pearsonr(x, y)
        r_p_text = textwrap.wrap("x mean: " + "{:.3g}".format(np.mean(x)), 30)
        r_p_text.extend(textwrap.wrap("y mean: " + "{:.2g}".format(np.mean(y)), 30))
        r_p_text.extend(
            textwrap.wrap(
                "Pearson’s correlation coefficient: " + "{:.2g}".format(r), 30
            )
        )
        r_p_text.extend(textwrap.wrap("2-taild p-value: " + "{:.2g}".format(p), 30))
        fig.text(
            sc_left + sc_width + space,
            sc_bottom + sc_height + space,
            "\n".join(r_p_text),
            fontsize=6,
        )
    ##########################
    # 平均とSEをプロット
    ##########################
    if avg is not False:
        min_x, max_x = ax.get_xlim()
        x_avg = np.arange(
            max(np.floor(np.min(x)), min_x),
            min(np.ceil(np.max(x)), max_x),
            avg,
            dtype=np.float64,
        )  # 平均を取る
        y_avg, y_se = [], []
        for i in x_avg:
            idx = (x < (i + avg)) * (x >= i)
            y_avg.append(np.average(y[idx]))
            # y_se.append(np.std(y[idx],ddof=1)/np.sqrt(y[idx].size)) #se
            y_se.append(np.std(y[idx], ddof=1))  # SD
        ax.errorbar(
            x_avg + avg * 0.5, y_avg, yerr=y_se, fmt="r.", ecolor="k", elinewidth=0.5
        )
    #############################
    # box prot
    ############################
    if box is not False:
        ax_b = ax.twiny()
        min_x, max_x = ax.get_xlim()
        x_avg = np.arange(min_x, max_x, box, dtype=np.float64)  # 平均を取る
        y_avg = []
        for i in x_avg:
            idx = (x < (i + box)) * (x >= i)
            y_avg.append(y[idx])
        whiskerprops = {"linestyle": "solid", "linewidth": -1, "color": "k"}
        ax_b.boxplot(
            y_avg,
            positions=(x_avg + box * 0.5),
            showmeans=True,
            meanline=True,
            whis="range",
            widths=box,
            whiskerprops=whiskerprops,
            showcaps=False,
        )
        ax_b.set_xticks([])  # x軸の調整
    # ここまで
    ###################
    if xticks is not False:
        ax.set_xticks(xticks)
    if yticks is not False:
        ax.set_yticks(yticks)
    if xticklabels != []:
        ax.set_xticklabels(xticklabels)
    if yticklabels != []:
        ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ###################
    # hisutoglam
    ###################
    ax_x = plt.axes([sc_left, sc_bottom + sc_height + space, sc_width, hst_height])
    ax_x.set_ylim(0, max_hist_x)
    ax_x.set_xlim(ax.get_xlim())
    ax_x.set_xticklabels([])
    ax_x.hist(x, bins=bin_hist_x, histtype="stepfilled", range=ax.get_xlim())
    if title is not False:
        plt.title(title, loc="right", fontsize=6)
    ax_y = plt.axes([sc_left + sc_width + space, sc_bottom, hst_height, sc_width])
    ax_y.set_ylim(ax.get_ylim())
    ax_y.set_xlim(0, max_hist_y)
    ax_y.set_yticklabels([])
    ax_y.hist(
        y,
        bins=bin_hist_y,
        histtype="stepfilled",
        range=ax.get_ylim(),
        orientation="horizontal",
    )
    if pdfpages is False:
        plt.savefig(save_file)
        plt.close()
    else:
        plt.savefig(pdfpages, format="pdf")
        plt.clf()
