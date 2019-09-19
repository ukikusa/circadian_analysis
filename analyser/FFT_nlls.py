# -*- coding: utf-8 -*-
"""fft_nllsをする．標準化には不偏標準偏差を使う"""

import os

import analyser.cos_models as cos_models
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

np.set_printoptions(precision=5, floatmode="fixed", suppress=True)


def data_trend(data, w):  # wはdata数
    """trandを求める

    Args:
        data: 列ごとに時系列データが入っているnumpy
        w: trandを取る範囲のデータ数．偶数なら+1する

    Returns:
        average, sd
        numpy np.float64
    """
    if w % 2 == 0:
        w = w + 1
    w2 = int(np.floor(w / 2))
    ws = np.arange(data.shape[0], dtype=np.int32)
    ws = ws - w2  # index 最初
    we = ws + w  # index 最後
    # 指定範囲が取れない場合は取れる範囲で取る
    ws[ws < 0] = 0
    we[we > data.shape[0]] = data.shape[0]
    move_sd = np.empty_like(data).astype(np.float64)
    move_avg = np.empty_like(move_sd)
    for i in range(data.shape[0]):
        move_sd[i] = np.std(data[ws[i] : we[i]], axis=0, ddof=1)
        move_avg[i] = np.average(data[ws[i] : we[i]], axis=0)
    return move_avg, move_sd


def data_norm(data, dt=60):
    # 24hでのdataのtrendを除去するversion
    data_avg, data_sd = data_trend(data, w=int(24 * 60 / dt))
    data_det = data - data_avg
    data_det_ampnorm = data_det / data_sd
    data_ampindex = data_sd / data_avg

    # normalize 平均を引いてSDで割る．
    data_det_norm = (data_det - np.average(data_det, axis=0)) / np.std(
        data_det, axis=0, ddof=1
    )
    data_norm = (data - np.average(data, axis=0)) / np.std(data, axis=0, ddof=1)
    return data_det, data_det_ampnorm


def fft_peak(data, s=0, e=24 * 3, dt=60, pdf_plot=False):
    """fftによる周期推定．範囲は両端を含む

    Args:
        data: numpy
        s: [time (h)] (default: {0})
        e: [time (h)] (default: {24 * 3})
        dt: [description] (default: {60})
        pdf_plot: [description] (default: {False})

    Returns:
        [description]
        [type]
    """
    dt_h = dt / 60
    data = data[int(s / dt_h) : int(e / dt_h) + 1]  # FFTに使うデータだけ．
    n = data.shape[0]
    time = np.arange(s, e + dt_h, dt_h)
    time = time - s
    # f = 1/dt_h*np.arange(int(n/2))/n  # 1時間あたりの頻度
    f = np.linspace(0, 1.0 / dt_h, n)
    # FFTアルゴリズム(CT)で一次元のn点離散フーリエ変換（DFT）
    fft_data = np.fft.fft(data, n=None, axis=0)  # axisは要検討 # norm="ortho"で正規化．よくわからん
    P2 = np.abs(fft_data) / n  # 振幅を合わせる．
    P1 = P2[0 : int(n / 2)]  # サンプリング頻度の半分しか有効じゃない
    P1[1:-1] = 2 * P1[1:-1]  # 交流成分を二倍．rのampspecとほぼ同じ
    P1[0] = 0
    # https://jp.mathworks.com/help/matlab/ref/fft.html
    fft_point = sp.signal.argrelmax(P1, order=1, axis=0)  # peakの場所
    fft_df = pd.DataFrame(index=[], columns=["sample", "amp", "f", "pha"])
    fft_df["sample"] = fft_point[1]
    fft_df["amp"] = P1[fft_point]
    fft_df["f"] = f[fft_point[0]]
    # fft_df['per'] = np.mod(np.angle(fft_data)[fft_point]+2*np.pi, 2*np.pi)
    # 複素数なので位相が出る
    fft_df["pha"] = np.angle(fft_data)[fft_point]
    # 複素数なので位相が出る
    fft_df = fft_df.sort_values(by=["sample", "amp"], ascending=[True, False])
    return fft_df, time, data


def start_plot(save_path, size_x=11.69, size_y=8.27):
    """A function that plots multiple graphs on one page."""
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


def fit_plot(
    pp, fig, ax, i, x, y, func, y_min=None, y_max=None, plt_x=5, plt_y=4, title=False
):
    # 1pageに対して，配置する graf の数．配置する graf の場所を指定．
    plt_n = plt_x * plt_y  # 一つのグラフへのプロット数
    i_mod = i % plt_n
    ax.append(fig.add_subplot(plt_x, plt_y, i_mod + 1))
    # プロット
    ax[i_mod].plot(x, y, linewidth=0, marker=".")
    # 軸の調整とか
    ax[i_mod].set_xlim(x[0], x[-1])  # x軸
    ax[i_mod].set_ylim(y_min, y_max)  # y軸
    ax[i_mod].set_xticks(np.arange(x[0], x[-1], 24))  # メモリ
    ax[i_mod].grid(
        which="major", axis="x", color="k", linestyle="dotted", lw=0.5
    )  # 縦の補助線
    ax[i_mod].set_title(title)
    ax[i_mod].tick_params(labelbottom=True, labelleft=True, labelsize=5)

    ###################
    # fittingのプロット
    ###################
    per_n = str(int(func.size / 3))
    ax[i_mod].plot(x, eval("cos_models.cos_model_" + per_n + "(x, *func)"), "-r", lw=1)
    if np.mod(i, plt_n) == plt_n - 1:
        pdf_save(pp)
        ax = []
    return ax


def cos_fit(data, s=0, e=24 * 3, dt=60, pdf_plot=False, tau_range=[16, 30], pdf=False):

    result_df = pd.DataFrame(
        index=[list(range(data.shape[1]))], columns=["amp", "tau", "pha", "rae"]
    )
    fft_df, time, data = fft_peak(data, s=s, e=e, dt=dt, pdf_plot=pdf_plot)
    fft_df = fft_df.rename(columns={"f": "tau"})
    fft_df["tau"] = 1 / fft_df["tau"]
    if pdf is not False:
        pp, fig, ax = start_plot(pdf, size_x=11.69, size_y=8.27)
        plt_x, plt_y = 5, 4
        plt_n = plt_x * plt_y
    for i in np.unique(fft_df["sample"]):  # data毎にループ
        p0, result, perr = [], [], []
        data_i, fft_df_i = (
            data[:, i],
            fft_df[fft_df["sample"] == i].reset_index(drop=True),
        )
        p0 = []
        for j in range(len(fft_df_i["sample"])):  # 推定した周期毎にフィッティング
            p0 = np.hstack(
                [p0, fft_df_i["amp"][j], fft_df_i["tau"][j], fft_df_i["pha"][j]]
            )  # fftで求めた初期値を用いる
            try:
                res, pcov = sp.optimize.curve_fit(
                    eval("cos_models.cos_model_" + str(j + 1)),
                    time,
                    data_i,
                    p0=p0,
                    ftol=1e-05,
                )
                per = np.sqrt(np.diag(pcov))  # 標準偏差
            except:
                print("tol = ではerrerが出てるよ")
                break
            p0[: len(res)] = res
            res = res.reshape([-1, 3])
            per = per.reshape([-1, 3])
            # 振幅のSDが振幅を超えたらだめ
            if np.min(np.abs(res[:, 0] / per[:, 0])) < 1:
                break
            else:
                result = res
                perr = per
            if j == 14:  # もっとしたければcos_modelsに関数を追加して．
                break
        if len(result) != 0:
            perr = perr[(result[:, 1] > tau_range[0]) * (result[:, 1] < tau_range[1])]
            result = result[
                (result[:, 1] > tau_range[0]) * (result[:, 1] < tau_range[1])
            ]
            result[result[:, 2] < 0, 2] = 2 * np.pi + result[result[:, 2] < 0, 2]
            result[result[:, 0] < 0, 0] = -result[result[:, 0] < 0, 0]
        if len(result) != 0:
            # RAEを求める．これでいいのだろうか
            UL = sp.stats.norm.interval(loc=result[0, 0], scale=perr[0, 0], alpha=0.95)
            result_df["rae"][i] = np.diff(UL) / result[0, 0] / 2
            result_df["amp"][i] = result[0, 0]
            result_df["tau"][i] = result[0, 1]
            result_df["pha"][i] = result[0, 2] / np.pi / 2 * (-1) + 1
            if pdf is not False:
                ax = fit_plot(
                    pp,
                    fig,
                    ax,
                    i,
                    time,
                    data_i,
                    result.flatten(),
                    y_min=None,
                    y_max=None,
                    plt_x=plt_x,
                    plt_y=plt_y,
                    title=i,
                )
    if pdf is not False:
        if np.mod(i, plt_n) != plt_n - 1:
            pdf_save(pp)
            plt.clf()
            pp.close()
    return result_df


if __name__ == "__main__":
    # os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))  # カレントディレクトリの設定
    os.chdir(
        os.path.join(
            "/hdd1",
            "Users",
            "kenya",
            "Labo",
            "keisan",
            "R_File",
            "190111_LDLL_CCA1_ALL",
            "data",
        )
    )
    dt = 20
    data_path = os.path.join("TimeSeries.txt")  # 解析データのパス
    df = pd.read_table(data_path)
    data = df.values

    data_det, data_det_ampnorm = data_norm(data, dt=20)

    np.savetxt("tmp.csv", data_det_ampnorm, delimiter=",")
    fft_nlls = cos_fit(
        data_det, s=48, e=120, dt=20, pdf_plot=False, tau_range=[10, 40], pdf="tmp.pdf"
    )
    fft_nlls_res = cos_fit(
        data_det_ampnorm,
        s=48,
        e=120,
        dt=20,
        pdf_plot=False,
        tau_range=[10, 40],
        pdf="tmp.pdf",
    )
    fft_nlls_res.to_csv("tmp.csv")
