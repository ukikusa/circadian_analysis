# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.optimize import curve_fit
from scipy import signal
from matplotlib.backends.backend_pdf import PdfPages
# 下２行は，リモートで操作している場合
# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt


def make_fig(x, y, save_path, peak=False, func=False, r=False, label=False, y_min=0, y_max=None, plt_x=5, plt_y=4, size_x=11.69, size_y=8.27):
    # yはたくさんある前提. peakを中心に前後rでフィティングをしたとする．
    # 28個の時系列折れ線グラフを一つのPDFファイルとして出力する．
    plt_n = plt_x * plt_y  # 一つのグラフへのプロット数
    # 文字サイズとかの設定
    plt.rcParams['font.size'] = 5
    plt.rcParams['font.family'] = 'sans-serif'
    # if os.path.exists(save_path) is False:
    #     os.makedirs(save_folder)
    #     print(save_folder+'を作成しました')
    if label is False:
        label = np.empty(y.shape[1])
        label[:] = False
    print('pdf作成を開始します')
    pp = PdfPages(save_path)
    fig = plt.figure(figsize=(size_x, size_y), dpi=100)
    ax = []
    ################ pdfの保存関数 ここから##############

    def pdf_save(pp):
        plt.tight_layout()  # レイアウト
        plt.savefig(pp, format='pdf')
        plt.clf()
        fig = plt.figure(figsize=(size_x, size_y), dpi=100)
    ################ ループ ##############
    for i in range(y.shape[1]):
        # 1pageに対して，配置する graf の数．配置する graf の場所を指定．
        i_mod = i % plt_n
        ax.append(fig.add_subplot(plt_x, plt_y, i_mod+1))
        # プロット
        ax[i_mod].plot(x, y[:, i], linewidth=0, marker='.')
        # 軸の調整とか
        ax[i_mod].set_xlim(left=0)  # x軸
        ax[i_mod].set_ylim(y_min, y_max)  # y軸
        ax[i_mod].set_xticks(np.arange(0, x[-1], 24) - x[0])  # メモリ
        ax[i_mod].grid(which='major', axis='x', color='k', linestyle='dotted', lw=0.5)  # 縦の補助線
        ax[i_mod].set_title(label[i])
        ax[i_mod].tick_params(labelbottom=True, labelleft=True, labelsize=5)
        ############## fittingのプロット #################
        if peak is not False:
            peak_i = peak[:, i]
            peak_i = peak_i[~np.isnan(peak_i)]
            for count, j in enumerate(peak_i):
                ax[i_mod].plot(x[int(j-r):int(j+r)], np.poly1d(func[count, i])(x[int(j-r): int(j+r)]), '-r', lw=1)
        ############## pdfの保存 #################
        if np.mod(i, plt_n) == plt_n-1:
            pdf_save(pp)
    ########## 残ったやつのPDFの保存 ############
    if np.mod(i, plt_n) != plt_n-1:
        pdf_save(pp)
    plt.clf()
    pp.close()


def moving_avg(data, avg=2):
    # dataはタイムシリーズ，dTは分，rangeはデータ数で指定
    # 移動平均を取るデータ数を決定．繰り上げ．
    # 以下移動平均
    a = np.ones(avg) / avg
    avg_data = np.convolve(data, a, 'valid')
#    time = np.arange(0, (avg_data.shape[0])*dT/60, dT/60) + avg/2*dT/60
    return avg_data


def peak_find(data, avg, p_range=6, fit_range=4, time=False, peak_v_range=0):
    # dataはdata,avgで移動平均．前後p_range分より高い点を暫定ピークに．その時の移動平均がpeak_avg．fit_rangeでフィッティング
    if avg != 1:  # 移動平均
        data_move = moving_avg(data, avg=avg)
        if len(time) != len(data_move):  # timeを移動平均に合わせる．
            time = time[int(avg / 2): data.shape[0] - int(avg / 2)]
        data = data_move  # アホなのでおまじない
    ############ ピークポイント抽出 ###############
    peak_tmp = signal.argrelmax(data, order=p_range)[0]
    peak_tmp = peak_tmp[np.where(peak_tmp < time.shape[0]-fit_range, 1, 0) * np.where(peak_tmp > fit_range + 1, 1, 0) == 1]  # fitting範囲が取れない場所を除く
    peak_tmp = peak_tmp[data[peak_tmp]/np.max(peak_tmp)>peak_v_range]
    if len(peak_tmp) == 0:  # ピークがないとき
        return [], [], [np.nan, np.nan, np.nan], [], []
    # 箱作り
    func = np.empty((len(peak_tmp), 3), dtype=np.float64)
    r2 = np.empty((len(peak_tmp)), dtype=np.float64)
    for count, i in enumerate(peak_tmp):  # ピークポイント毎にfiting
        fit = np.polyfit(time[i - fit_range: i + fit_range + 1], data[i - fit_range: i + fit_range + 1], deg=2, full=True)  # fitting本体．
        func[count] = fit[0]
        r2[count] = 1 - fit[1]/np.var(data[i - fit_range: i + fit_range + 1], ddof=fit_range*2)
    peak_t = -func[:, 1] * 0.5 / func[:, 0]  # peak時間
    peak_v = func[:, 2] - pow(func[:, 1], 2) * 0.25 / func[:, 0]  # peak時の値
    j = np.where(func[:, 0] < 0, 1, 0) * np.where(peak_t < np.max(time), 1, 0) * np.where(peak_t > 0, 1, 0)
    return peak_t[j == 1], peak_v[j == 1], func[j == 1], r2[j == 1], peak_tmp[j == 1]


def make_phase(peak_time, n=False, dT=60, time=False):
    peak_time = np.array(peak_time)
    peak_time = peak_time[~np.isnan(peak_time)]
    if time is False:
        time = np.arange(n)*dT/60
    phase = np.empty(len(time), dtype=np.float64)
    phase[:] = np.nan
    period = np.copy(phase)
    if peak_time != []:
        peak_idx = np.searchsorted(time, peak_time)  # timeに当たるindex
        for i in range(peak_time.shape[0] - 1):
            # ピークとピークの間を0-1に均等割．
            phase[peak_idx[i]:peak_idx[i+1]] = (time[peak_idx[i]:peak_idx[i + 1]] - peak_time[i]) / (peak_time[i + 1] - peak_time[i])
            period[peak_idx[i]:peak_idx[i+1]] = peak_time[i+1]-peak_time[i]
    return phase, period


def phase_analysis(data, avg, dT=60, p_range=6, fit_range=4, offset=0, time=False, peak_v_range=0):
    # peak時間，値，Phase，フィッティングのr2値，フィッティングに使ったインデックス(移動平均取ったあと)．
    if time is False:
        time = np.arange(data.shape[0], dtype=np.float64) * dT / 60 + offset
    time_move = time[int(avg / 2): data.shape[0] - int(avg / 2)]
    ################3 peakや時間を出力する箱を作る． ###############
    data_phase, data_period = np.empty_like(data, dtype=np.float64), np.empty_like(data, dtype=np.float64)
    peak_t = np.zeros((int(data.shape[0] / p_range), data.shape[1]), dtype=np.float64)
    peak_v, r2, peak_point = np.zeros_like(peak_t), np.zeros_like(peak_t), np.zeros_like(peak_t)
    func = np.zeros((int(data.shape[0] / p_range), data.shape[1], 3))
    # それぞれの時系列に対して二次関数フィッティングを行う．
    for i in range(data.shape[1]):
        # peakを推定する．
        fit = peak_find(
            data[:, i], time=time, avg=avg, p_range=p_range, fit_range=fit_range,peak_v_range=peak_v_range)
        peak_t[0:len(fit[0]), i], peak_v[0:len(fit[1]), i], r2[0:len(fit[1]), i], func[0:len(fit[1]), i], peak_point[0:len(fit[1]), i] = fit[0], fit[1], fit[3], fit[2], fit[4]  # 馬鹿な代入…
        data_phase[:, i], data_period[:, i] = make_phase(fit[0], data.shape[0], dT=dT, time=time)
    idx = np.nonzero(np.sum(peak_point, axis=1))  # いらんとこ消す
    idx_0 = peak_point == 0
    peak_t[idx_0], peak_v[idx_0], r2[idx_0], peak_point[idx_0], func[idx_0] = np.nan, np.nan, np.nan, np.nan, np.nan
    return peak_t[idx], peak_v[idx], data_phase, r2[idx], peak_point[idx], func[idx], data_period

