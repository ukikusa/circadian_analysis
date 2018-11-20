# -*- coding: utf-8 -*-
"""Function group for finding peaks from time series data."""

import sys

import numpy as np

from scipy import signal


def moving_avg(data, avg=2):
    """A function for obtaining moving average."""
    a = np.ones(avg) / avg
    avg_data = np.convolve(data, a, 'valid')
    return avg_data


def peak_find(data, p_tmp, avg=1, f_range=9, time=False):
    """二次関数フィッティングを行う．

    Args:
        data: {np.array. 1} 1次元のArray
        p_tmp: {list} Fitting中心の座標
        avg: {int} fitting前に移動平均を取るか．多分不要． (default: {1})
        f_range: f_range: {int} 前後それぞれf_rangeのデータを用いて推定をする.  (default: {9})
        time: 時間軸を入れる (default: {False})

    Returns:
        頂点のX座標, 頂点のY座標, 二次関数の係数, r2値, Fitingの中心のIndex
        [type]
    """
    if time is False:
        time = np.arange(data.shape[0])
    n = data.shape[0] - avg + 1
    if avg != 1:  # 移動平均
        data = moving_avg(data, avg=avg)
        p_tmp = p_tmp - avg % 2
        time = time[(avg % 2): (avg % 2) + n]
    if len(time) != len(data):
        sys.exit("Timeの長さとDataの長さを一致させてください．")
    ####################
    # ピークポイント抽出
    ####################
    p_tmp = p_tmp[np.where(p_tmp < n - f_range, 1, 0) * np.where(p_tmp > f_range + 1, 1, 0) == 1]
    # fitting範囲が取れない場所を除く
    tmp_n = len(p_tmp)
    if tmp_n == 0:  # ピークがないとき
        return [], [], [np.nan, np.nan, np.nan], [], []
    # 箱作り
    func = np.empty((tmp_n, 3), dtype=np.float64)
    r2 = np.empty(tmp_n, dtype=np.float64)
    ###################################
    # fitting
    ###################################
    for count, i in enumerate(p_tmp):
        i_f, i_e = i - f_range, i + f_range + 1
        fit = np.polyfit(time[i_f: i_e], data[i_f: i_e], deg=2, full=True)
        func[count] = fit[0]
        r2[count] = (1 - fit[1] / np.var(data[i_f: i_e], ddof=f_range * 2))
    peak_t = -func[:, 1] * 0.5 / func[:, 0]  # peak時間
    peak_v = func[:, 2] - pow(func[:, 1], 2) * 0.25 / func[:, 0]  # peak時の値
    j = np.where(func[:, 0] < 0, 1, 0) * np.where(peak_t < time[p_tmp + f_range], 1, 0) * np.where(time[p_tmp - f_range] > 0, 1, 0)
    return peak_t[j == 1], peak_v[j == 1], func[j == 1], r2[j == 1], p_tmp[j == 1]


def make_phase(peak_time, n=168, dt=60, time=False):
    """A function that creates a list of phases from the peak list.
    
    Args:
        peak_time: peak時間 (hour)
        n: 出力するデータ数 (default: {168})
        dt: データ間隔 (default: {60})
        time: {list} 出力するデータの時間.dt,nに優先 (default: {False})

    Returns:
        [phase, img]
        [np.array. 位相は0-1.]
    """
    peak_time = np.array(peak_time)
    peak_time = peak_time[~np.isnan(peak_time)]
    if time is False:
        time = np.arange(n) * dt / 60
    phase = np.empty(len(time), dtype=np.float64)
    phase[:] = np.nan
    period = np.copy(phase)
    if peak_time != []:
        peak_idx = np.searchsorted(time, peak_time)  # timeに当たるindex
        for i in range(peak_time.shape[0] - 1):
            # ピークとピークの間を0-1に均等割．
            phase[peak_idx[i]:peak_idx[i + 1]] = (time[peak_idx[i]:peak_idx[i + 1]] - peak_time[i]) / (peak_time[i + 1] - peak_time[i])
            period[peak_idx[i]:peak_idx[i + 1]] = peak_time[i + 1] - peak_time[i]
    return phase, period


def phase_analysis(data, avg, dt=60, p_range=12, f_avg=1, f_range=5, offset=0, time=False):
    """データ群に対して二次関数フィッティングを行い，位相等のデータを出力する.

    Args:
        data: {np.array. dim 2} 発光量のデータ．i列をi個目の時系列だと認識．
        avg: {int} 移動平均を取るデータ数．奇数が望ましい．
        dt: {int} minute. 時間軸作成のため (default: {60})
        p_range: {int} 前後それぞれp_rangeよりも値が高い点をピークとみなす. (default: {12})
        f_range: {int} 前後それぞれf_rangeのデータを用いて推定をする. (default: {5})
        offset: {int} hour. 開始時間. (default: {0})
        time: {list or np.array} 時間軸．dt, offsetと共存しない． (default: {False})

    Returns:
        [[peak時間]，[Peakの値]，[Phase list]，[r2]，[p_rangeによりpeakのindex(移動平均後)]
        [np.arrayの入ったリスト]
    """
    t_n = data.shape[0]
    d_n = data.shape[1]
    if time is False:
        time = np.arange(data.shape[0], dtype=np.float64) * dt / 60 + offset
    p_tmp = signal.argrelmax(data, order=p_range, axis=0)  # 周りより値が大きい点抽出
    p_tmp_n = np.max(np.bincount(p_tmp[0]))
    ###############
    # peakや時間を出力する箱を作る.
    ###############
    d_theta = np.empty_like(data, dtype=np.float64)
    d_tau = np.empty_like(data, dtype=np.float64)
    peak_t = np.zeros((p_tmp_n, d_n), dtype=np.float64)
    peak_v, r2, peak_point = np.zeros_like(peak_t), np.zeros_like(peak_t), np.zeros_like(peak_t)
    func = np.zeros((p_tmp_n, data.shape[1], 3))
    # それぞれの時系列に対して二次関数フィッティングを行う．
    for i in range(d_n):
        # peakを推定する．
        p_tmp_i = p_tmp[0][p_tmp[1] == i]
        if len(p_tmp_i) <= 1:
            d_theta[:, i], d_tau[:, i] = np.nan, np.nan
        else:
            fit = peak_find(data[:, i], p_tmp=p_tmp_i, avg=f_avg, f_range=f_range, time=time)
            peak_t[:len(fit[0]), i] = fit[0]
            peak_v[:len(fit[1]), i] = fit[1]
            r2[:len(fit[1]), i] = fit[3]
            func[:len(fit[1]), i] = fit[2]
            peak_point[:len(fit[1]), i] = fit[4]
            d_theta[:, i], d_tau[:, i] = make_phase(fit[0], t_n, dt=dt, time=time)
    idx = np.nonzero(np.sum(peak_point, axis=1))  # いらんとこ消す
    idx_0 = peak_point == 0
    peak_t[idx_0], peak_v[idx_0], r2[idx_0], peak_point[
        idx_0], func[idx_0] = np.nan, np.nan, np.nan, np.nan, np.nan
    return peak_t[idx], peak_v[idx], d_theta, r2[idx], peak_point[idx], func[idx], d_tau
