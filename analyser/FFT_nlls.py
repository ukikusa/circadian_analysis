# -*- coding: utf-8 -*-
'''fft_nllsをする．標準化には不偏標準偏差を使う'''

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
np.set_printoptions(precision=5, floatmode='fixed', suppress=True)


def data_trend(data, w):  # wはdata数
    '''trandを求める
    
    Args:
        data: 列ごとに時系列データが入っているnumpy
        w: trandを取る範囲のデータ数．偶数なら+1する
    
    Returns:
        average, sd
        numpy np.float64
    '''
    if w % 2 == 0:
        w = w + 1
    w2 = int(np.floor(w / 2))
    ws = np.arange(data.shape[0], dtype=np.int32)
    ws = ws - w2  # index 最初
    we = ws + w  # index 最後
    # 指定範囲が取れない場合は取れる範囲で取る
    ws[ws<0] = 0
    we[we>data.shape[0]] = data.shape[0]
    move_sd = np.empty_like(data).astype(np.float64)
    move_avg = np.empty_like(move_sd)
    for i in range(data.shape[0]):
        move_sd[i] = np.std(data[ws[i]:we[i]], axis=0, ddof=1)
        move_avg[i] = np.average(data[ws[i]:we[i]], axis=0)
    return move_avg, move_sd


def data_norm(data, dt=60):
    # 24hでのdataのtrendを除去するversion
    data_avg, data_sd = data_trend(data, w=int(24 * 60 / dt))
    data_det = data - data_avg
    data_det_ampnorm = data_det/data_sd
    data_ampindex = data_sd/data_avg

    # normalize 平均を引いてSDで割る．
    data_det_norm = (data_det - np.average(data_det, axis=0))/np.std(data_det, axis=0, ddof=1)
    data_norm = (data - np.average(data, axis=0))/np.std(data, axis=0, ddof=1)
    return data_det, data_det_ampnorm


def fft_peak(data, s=0, e=24 * 3, dt=60, pdf_plot=False):
    '''fftによる周期推定．範囲は両端を含む
    
    Args:
        data: numpy
        s: [time (h)] (default: {0})
        e: [time (h)] (default: {24 * 3})
        dt: [description] (default: {60})
        pdf_plot: [description] (default: {False})
    
    Returns:
        [description]
        [type]
    '''
    dt_h = dt / 60
    data = data[int(s / dt_h):int(e / dt_h) + 1]  # FFTに使うデータだけ．
    n = data.shape[0]
    time = np.arange(s, e + dt_h, dt_h)
    # f = 1/dt_h*np.arange(int(n/2))/n  # 1時間あたりの頻度
    f = np.linspace(0, 1.0 / dt_h, n)
    # FFTアルゴリズム(CT)で一次元のn点離散フーリエ変換（DFT）
    fft_data = np.fft.fft(data, n=None, axis=0)  # axisは要検討 # norm="ortho"で正規化．よくわからん
    P2 = np.abs(fft_data) / n  # 振幅を合わせる．
    P1 = P2[0:int(n / 2)]  # サンプリング頻度の半分しか有効じゃない
    P1[1:-1] = 2 * P1[1:-1]  # 交流成分を二倍．rのampspecとほぼ同じ
    P1[0] = 0
    # https://jp.mathworks.com/help/matlab/ref/fft.html
    fft_point = sp.signal.argrelmax(P1, order=1, axis=0)  # peakの場所
    fft_df = pd.DataFrame(index=[], columns=['sample', 'amp', 'f', 'pha'])
    fft_df['sample'] = fft_point[1]
    fft_df['amp'] = P1[fft_point]
    fft_df['f'] = f[fft_point[0]]
    # fft_df['per'] = np.mod(np.angle(fft_data)[fft_point]+2*np.pi, 2*np.pi) 
    # 複素数なので位相が出る
    fft_df['pha'] = np.angle(fft_data)[fft_point]/np.pi
    # 複素数なので位相が出る
    fft_df = fft_df.sort_values(by=['sample', 'amp'], ascending=[True, False])
    if pdf_plot is not False:
        fig = plt.figure(figsize=(6, 4), dpi=100)
        plt.plot(f, P1[:, 0])
        # plt.plot( 1/dt_h*np.arange(int(n))/n, P2[:, 0])
        plt.show()
        plt.clf()
        plt.plot(time, phase_fft)
        plt.show()
        plt.clf()
    print(fft_df)
    return fft_df, time, data


def cos_fit(data, s=0, e=24 * 3, dt=60, pdf_plot=False, tau_range=[16, 30]):

    result_df = pd.DataFrame(index=[list(range(data.shape[1]))], columns=['amp', 'tau', 'pha', 'rae'])
    dt_h = dt / 60
    fft_df, time, data = fft_peak(data, s=s, e=e, dt=dt, pdf_plot=pdf_plot)
    fft_df = fft_df.rename(columns={'f': 'tau'})
    fft_df['tau'] = 1 / fft_df['tau']

    def cos_model(time, amp, tau, pha, offset):  # fittingのモデル
        return amp * np.cos(2 * np.pi * (time / tau) + np.pi * pha) + offset
    for i in np.unique(fft_df['sample']):  # data毎にループ
        result = []
        perr = []
        data_i, fft_df_i = data[:, i], fft_df[fft_df['sample'] == i].reset_index(drop=True)
        for j in range(len(fft_df_i['sample'])):  # 推定した周期毎にフィッティング
            p0 = np.array([fft_df_i['amp'][j], fft_df_i['tau'][j], fft_df_i['pha'][j], np.average(data_i)])  # fftで求めた初期値を用いる
            try:
                result_t, pcov = sp.optimize.curve_fit(cos_model, time, data_i, p0=p0)  # , bounds=[[0, np.inf]]) # fitting
                perr.append(list(np.sqrt(np.diag(pcov))))  # 標準偏差
                data_i = data_i - cos_model(time, result_t[0], result_t[1], result_t[2], result_t[3])
                # fitting結果を引く．
                result.append(result_t)  # 結果
            except:
                print('tol = ではerrerが出てるよ')
                break
            # if np.min(np.abs(np.array(result)[:,0]/result_t[0]))<1: # 振幅が前の結果より大きなったらsそこまで
            #   break
        result, perr = np.array(result), np.array(perr)
        perr = perr[(result[:, 1] > tau_range[0]) * (result[:, 1] < tau_range[1])]
        result = result[(result[:, 1] > tau_range[0]) * (result[:, 1] < tau_range[1])]
        result[result[:, 0] < 0, 2] = np.pi - result[result[:, 0] < 0, 2]
        result[result[:, 0] < 0, 0] = -result[result[:, 0] < 0, 0]

        try:
            UL = sp.stats.norm.interval(loc=result[0, 0], scale=perr[0, 0], alpha=0.95)
        except:
                break
        result_df['rae'][i] = np.diff(UL) / result[0, 0] / 2
        result_df['amp'][i] = result[0, 0]
        result_df['tau'][i] = result[0, 1]
        result_df['pha'][i] = result[0, 2]
    return result_df

if __name__ == '__main__':
    # os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))  # カレントディレクトリの設定
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'R_File', '190111_LDLL_CCA1_ALL', 'data'))
    dt=20
    data_path = os.path.join('TimeSeries.txt')  # 解析データのパス
    df = pd.read_table(data_path)
    data = df.values
    data_det, data_det_ampnorm = data_norm(data, dt=20)

    np.savetxt("tmp.csv", data_det_ampnorm, delimiter=",")
    a = cos_fit(data_det, s=48, e=120, dt=20, pdf_plot=False, tau_range=[10, 40])
    print(a)
    a.to_csv("tmp.csv")
