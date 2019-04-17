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


def amp_analysis(data, h_range=24 * 3):
    range_2 = int(h_range / 2)
    data = data.astype(np.float64)
    cv = np.full_like(data, np.nan)
    data[data == 0] = np.nan
    sd = np.copy(cv)
    n = data.shape[0]
    n_e = n - range_2
    for i in range(range_2, n_e):
        data_i = data[i - range_2:i + range_2]
        sd_i = np.std(data_i, axis=0)
        sd[i] = sd_i
        cv[i] = sd_i / np.average(data_i, axis=0)
    return cv, sd


def peak_find(data, p_tmp, avg=1, f_range=9, time=False, r2_cut=0):
    """二次関数フィッティングを行う.

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
    j = np.where(func[:, 0] < 0, 1, 0) * np.where(peak_t < time[p_tmp + f_range], 1, 0) * np.where(time[p_tmp - f_range] > 0, 1, 0) * np.where(r2 >= r2_cut, 1, 0)
    return peak_t[j == 1], peak_v[j == 1], func[j == 1], r2[j == 1], p_tmp[j == 1]


def peak_cut(peak_t, peak_v, func, r2, p_tmp, min_tau=16, max_tau=32):
    """不要なピークをカットする."""
    dic = {'peak_v': peak_v, 'func': func, 'p_tmp': p_tmp}

    def edege_long_del(peak_t, r2, max_tau=32, **dic):
        """"両はじに長い周期があったらそれを消す."""
        tau = np.diff(peak_t)
        big_idx = tau > max_tau
        if len(tau) == 1 and ~big_idx[0]:
            dic = {k: [] for (k, v) in dic.items()}
            return [], [], dic
        if big_idx[0] is True:
            peak_t, r2, big_idx = peak_t[1:], r2[1:], big_idx[1:]
            dic = {k: v[1:] for (k, v) in dic.items()}
            return peak_t, r2, dic
        if big_idx[-1] is True:
            peak_t, r2 = peak_t[:-1], r2[:-1]
            dic = {k: v[:-1] for (k, v) in dic.items()}
        return peak_t, r2, dic

    def short_tau_peak_del(peak_t, r2, min_tau=16, max_tau=32, **dic):
        """周期が短すぎるとき，ピークを一つ削除.

        両側のいずれのかの周期が16以下かつ，両側の周期を足したら32以下になるPeakを一つ削除．一番R2値が小さいのもの．
        """
        tau = np.diff(peak_t)  # 周期
        tau_2 = tau[:-1] + tau[1:]  # ピークを消したときの周期
        short_tau = tau <= min_tau  # 周期が短すぎるところのIndex
        short_tau_2 = short_tau[:-1] + short_tau[1:]
        del_opt = short_tau_2 * (tau_2 <= max_tau)
        if np.sum(del_opt) == 0:
            return peak_t, r2, dic
        # 削除する対象がなければ終わる
        ##############################
        r2_tmp = np.ones_like(r2)
        r2_tmp[1:-1][del_opt] = r2[1:-1][del_opt]
        del_idx = np.argmin(r2_tmp)
        peak_t = np.delete(peak_t, obj=del_idx)
        r2 = np.delete(r2, obj=del_idx)
        dic = {k: np.delete(v, obj=del_idx, axis=0) for (k, v) in dic.items()}
        return peak_t, r2, dic

    ###################
    # ここまで関数の定義
    ####################
    if len(peak_t) <= 1:
        return [], [], [], [], []
    for i in range(len(peak_t)):
        peak_t, r2, dic = short_tau_peak_del(peak_t, r2, **dic)
        if len(peak_t) <= 1:
            return [], [], [], [], []
        peak_t, r2, dic = edege_long_del(peak_t, r2, **dic)
        if len(peak_t) <= 1:
            return [], [], [], [], []
        if i == len(peak_t):
            return peak_t, dic['peak_v'], dic['func'], r2, dic['p_tmp']
    return peak_t, dic['peak_v'], dic['func'], r2, dic['p_tmp']


def make_phase(peak_time, n=168, dt=60, time=False, r2=False, min_tau=16, max_tau=32):
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
    no_nan = ~np.isnan(peak_time)
    peak_time = peak_time[no_nan]
    if time is False:
        time = np.arange(n) * dt / 60
    phase = np.empty(len(time), dtype=np.float64)
    phase[:] = np.nan
    tau = np.copy(phase)
    if peak_time != []:
        peak_idx = np.searchsorted(time, peak_time)  # timeに当たるindex
        for i in range(peak_time.shape[0] - 1):
            # ピークとピークの間を0-1に均等割．
            tai_i = peak_time[i + 1] - peak_time[i]
            if tai_i <= max_tau and tai_i >= min_tau:
                phase[peak_idx[i]:peak_idx[i + 1]] = (time[peak_idx[i]:peak_idx[i + 1]] - peak_time[i]) / (peak_time[i + 1] - peak_time[i])
                tau[peak_idx[i]:peak_idx[i + 1]] = peak_time[i + 1] - peak_time[i]
    return phase, tau


def phase_analysis(data, avg, dt=60, p_range=12, f_avg=1, f_range=5, offset=0, time=False, r2_cut=False, min_tau=16, max_tau=32):
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
        [[頂点のx]，[頂点のy]，[Phase list]，[r2]，[Fittingのindex], [Fitingの係数], [tau_list]]
        [np.arrayのリスト]
    """
    t_n = data.shape[0]
    d_n = data.shape[1]
    if time is False:
        time = np.arange(data.shape[0], dtype=np.float64) * dt / 60 + offset
    p_tmp = signal.argrelmax(data, order=p_range, axis=0)  # 周りより値が大きい点抽出
    p_tmp_n = np.max(np.bincount(p_tmp[1]))
    ###############
    # peakや時間を出力する箱を作る.
    ###############
    d_theta = np.empty_like(data, dtype=np.float64)
    d_tau = np.empty_like(data, dtype=np.float64)
    print(p_tmp)
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
            peak_t_i, peak_v_i, func_i, r2_i, p_tmp_i = peak_find(data[:, i], p_tmp=p_tmp_i, avg=f_avg, f_range=f_range, time=time, r2_cut=r2_cut)
            fit = peak_cut(peak_t_i, peak_v_i, func_i, r2_i, p_tmp_i, min_tau=min_tau, max_tau=max_tau)

            if len(fit[0]) <= 1:
                d_theta[:, i], d_tau[:, i] = np.nan, np.nan
            else:
                peak_t[:len(fit[0]), i] = fit[0]
                peak_v[:len(fit[1]), i] = fit[1]
                r2[:len(fit[1]), i] = fit[3]
                func[:len(fit[1]), i] = fit[2]
                peak_point[:len(fit[1]), i] = fit[4]
                d_theta[:, i], d_tau[:, i] = make_phase(fit[0], t_n, dt=dt, time=time, r2=fit[3], min_tau=min_tau, max_tau=max_tau)
    idx = np.nonzero(np.sum(peak_point, axis=1))  # いらんとこ消す
    idx_0 = peak_point == 0
    peak_t[idx_0], peak_v[idx_0], r2[idx_0], peak_point[
        idx_0], func[idx_0] = np.nan, np.nan, np.nan, np.nan, np.nan
    return peak_t[idx], peak_v[idx], d_theta, r2[idx], peak_point[idx], func[idx], d_tau
