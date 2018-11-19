# -*- coding: utf-8 -*-
"""Function group for finding peaks from time series data."""

import numpy as np

from scipy import signal


def moving_avg(data, avg=2):
    """A function for obtaining moving average."""
    a = np.ones(avg) / avg
    avg_data = np.convolve(data, a, 'valid')
    return avg_data


def peak_find(data, avg, p_range=13, f_range=9, time=False):
    """"A function that obtains the peak from the time series data by quadratic function fitting."""
    # 前後p_rangeの真ん中で最も高い値を持つ点を中心に前後f_rangeでフィティング
    n = data.shape[0] - avg + 1
    if avg != 1:  # 移動平均
        data_move = moving_avg(data, avg=avg)
        if len(time) != len(data_move):  # timeを移動平均に合わせる．
            time = time[(avg % 2): (avg % 2) + n]
    else:
        data = data_move  # アホなのでおまじない
    ####################
    # ピークポイント抽出
    ####################
    peak_tmp = signal.argrelmax(data, order=p_range)[0]
    peak_tmp = peak_tmp[np.where(peak_tmp < n - f_range, 1, 0)
                        * np.where(peak_tmp > f_range + 1, 1, 0) == 1]
    # fitting範囲が取れない場所を除く
    tmp_n = len(peak_tmp)
    if tmp_n == 0:  # ピークがないとき
        return [], [], [np.nan, np.nan, np.nan], [], []
    # 箱作り
    func = np.empty((tmp_n, 3), dtype=np.float64)
    r2 = np.empty(tmp_n, dtype=np.float64)
    ###################################
    # fitting
    ###################################
    for count, i in enumerate(peak_tmp):
        i_f, i_e = i - f_range, i + f_range + 1
        fit = np.polyfit(time[i_f: i_e], data[i_f: i_e], deg=2, full=True)
        func[count] = fit[0]
        r2[count] = (1 - fit[1] / np.var(data[i_f: i_e], ddof=f_range * 2))
    peak_t = -func[:, 1] * 0.5 / func[:, 0]  # peak時間
    peak_v = func[:, 2] - pow(func[:, 1], 2) * 0.25 / func[:, 0]  # peak時の値
    j = np.where(func[:, 0] < 0, 1, 0) * np.where(peak_t < time[peak_tmp + f_range], 1, 0) * np.where(time[peak_tmp - f_range] > 0, 1, 0)
    return peak_t[j == 1], peak_v[j == 1], func[j == 1], r2[j == 1], peak_tmp[j == 1]


def make_phase(peak_time, n=False, dt=60, time=False):
    """"A function that creates a list of phases from the peak list."""
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


def phase_analysis(data, avg, dt=60, p_range=6, f_range=4, offset=0, time=False):
    """データ群に対して二次関数フィッティングを行い，位相等のデータを出力する.

    Args:
        data: {np.array. dim 2} 発光量のデータ．i列をi個目の時系列だと認識．
        avg: {int} 移動平均を取るデータ数．奇数が望ましい．
        dt: {int} minute. 時間軸作成のため (default: {60})
        p_range: {int} 前後それぞれp_rangeよりも値が高い点をピークとみなす. (default: {6})
        f_range: {int} 前後それぞれf_rangeのデータを用いて推定をする. (default: {4})
        offset: {int} (default: {0})
        time: {list or np.array} 時間軸．dt, offsetと共存しない． (default: {False})

    Returns:
        [[peak時間]，[Peakの値]，[Phase list]，[r2]，[p_rangeによりpeakのindex(移動平均後)]
        [np.arrayの入ったリスト]
    """
    if time is False:
        time = np.arange(data.shape[0], dtype=np.float64) * dt / 60 + offset
    ###############
    # peakや時間を出力する箱を作る.
    ###############
    data_phase, data_period = np.empty_like(data, dtype=np.float64), np.empty_like(data, dtype=np.float64)
    peak_t = np.zeros((int(data.shape[0] / p_range), data.shape[1]), dtype=np.float64)
    peak_v, r2, peak_point = np.zeros_like(peak_t), np.zeros_like(peak_t), np.zeros_like(peak_t)
    func = np.zeros((int(data.shape[0] / p_range), data.shape[1], 3))
    # それぞれの時系列に対して二次関数フィッティングを行う．
    for i in range(data.shape[1]):
        # peakを推定する．
        fit = peak_find(data[:, i], time=time, avg=avg, p_range=p_range, f_range=f_range)
        peak_t[0:len(fit[0]), i], peak_v[0:len(fit[1]), i], r2[0:len(fit[1]), i], func[0:len(fit[1]), i], peak_point[0:len(fit[1]), i] = fit[0], fit[1], fit[3], fit[2], fit[4]  # 馬鹿な代入…
        data_phase[:, i], data_period[:, i] = make_phase(fit[0], data.shape[0], dt=dt, time=time)
    idx = np.nonzero(np.sum(peak_point, axis=1))  # いらんとこ消す
    idx_0 = peak_point == 0
    peak_t[idx_0], peak_v[idx_0], r2[idx_0], peak_point[
        idx_0], func[idx_0] = np.nan, np.nan, np.nan, np.nan, np.nan
    return peak_t[idx], peak_v[idx], data_phase, r2[idx], peak_point[idx], func[idx], data_period
