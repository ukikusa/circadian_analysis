# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
import pandas as pd
import scipy as sp


def fft_peak(data, s=0, e=24 * 3, dt=60, pdf_plot=True):
    # sとeの単位は時間
    dt_h = dt / 60
    data = data[int(s / dt_h):int(e / dt_h) + 1]  # FFTに使うデータだけ．
    n = data.shape[0]
    time = np.arange(s, e, dt_h)
    f = 1 / dt_h * np.arange(int(n / 2) + 1) / n  # 1時間あたりの頻度
    # FFTアルゴリズム(CT)で一次元のn点離散フーリエ変換（DFT）
    fft_data = np.fft.fft(data, n=None, axis=0)  # axisは要検討 # norm="ortho"で正規化．よくわからん
    P2 = np.abs(fft_data) / n  # 振幅を合わせる．
    P1 = P2[0:int(n / 2) + 1]  # サンプリング頻度の半分しか有効じゃない
    P1[1:-1] = 2 * P1[1:-1]  # 交流成分を二倍．rのampspecとほぼ同じ
    P1[0] = 0
    # https://jp.mathworks.com/help/matlab/ref/fft.html
    fft_point = sp.signal.argrelmax(P1, order=1)  # peakの場所
    fft_df = pd.DataFrame(index=[], columns=['sample', 'amp', 'f', 'per'])
    fft_df['sample'] = fft_point[1]
    fft_df['amp'] = P1[fft_point]
    fft_df['f'] = f[fft_point[0]]
    fft_df['per'] = np.angle(fft_data)[fft_point]
    print(fft_df, np.angle(fft_data))
    peak_time = peak_fft * dt_h + s
    phase_fft = np.angle(fft_data)
    print(P1.shape, peak_fft)
    if pdf_plot is not False:
        fig = plt.figure(figsize=(6, 4), dpi=100)
        plt.plot(f, P1[:, 0])
        # plt.plot( 1/dt_h*np.arange(int(n))/n, P2[:, 0])
        plt.show()
        plt.clf()
        plt.plot(time, phase_fft)
        plt.show()
        plt.clf()
    return peak_fft, peak_time


def cos_fit(data, dt):
    model = "y ~ cons"


if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))  # カレントディレクトリの設定
    data_path = os.path.join('')  # 解析データのパス
    data = np.empty((73, 3), dtype=np.float64)
    sin24 = np.sin(np.linspace(0, 3 * 2 * np.pi, num=73))
    sin12 = np.sin(np.linspace(0, 6 * 2 * np.pi, num=73))
    data[:, 0] = sin24
    data[:, 1] = sin24 * 2
    data[:, 2] = sin24 * 2 + sin12
    fft_peak(data, s=0, e=72)
