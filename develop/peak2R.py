# -*- coding: utf-8 -*-

import numpy as np
import os
import matplotlib.pyplot as plt
import peak_analysis as pa
import pandas as pd


def phase2R(phase_data):
    # phaseのデータを投げたら同期率Rと位相のリストを返す関数.2次元配列のみ対応．
    # オイラー展開（でいいのかな）するよ
    euler_data = np.exp(1j * phase_data[np.logical_xor(phase_data != -1, np.isnan(phase_data))] * 2 * np.pi)
    # 余裕がアレば，位相の画像返してあげてもいいね．
    R = np.mean(euler_data)
    return R, euler_data


def phase2Rs(data):
    r, number, euler_datas = [], [], []
    for i in range(data.shape[0]):
        if data.ndim == 3:
            data_tmp = data[i, :, :]
        elif data.ndim == 2:
            data_tmp = data[i, :]
        R, euler_data = phase2R(data_tmp)
        euler_datas.append(np.sum(euler_data))
        r.append(np.abs(R))
        number.append(np.size(euler_data))
    return r, number, euler_datas


def r_plt(x, r, pdf_save):
    plt.rcParams['font.size'] = 11  # 文字サイズ
    fig = plt.figure(figsize=(8, 6), dpi=100)  # 作図開始．PDFサイズ
    ax = fig.add_subplot(111)
    ax.plot(x, r)  # プロット
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)  # y軸整形
    ax.set_xticks(np.arange(0, x[-1], 24) - x[0])  # メモリ
    ax.vlines(np.arange(0, x[-1], 24), 0, 1, colors="k", linestyle="dotted", lw=0.5)  # 縦線
    ax.tick_params(labelbottom=True, labelleft=True, labelsize=5)
    ax.set_xlabel("time")  # x軸ラベル
    plt.savefig((os.path.join(save_folder, 'r.pdf')))
    plt.close()
    return 0


def peak2r(peak_folder, save_folder, pdf=False, dT=60, n=200):
    peak_data = pd.read_csv(peak_folder, dtype=np.float64, index_col=False)  # 読み込み
    peak = np.array(peak_data.values)  # numpyに
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    time = np.arange(n)*dT/60
    phase = np.empty((len(time), peak.shape[1]))
    for i in range(peak.shape[1]):
        phase[:, i], period = pa.make_phase(peak[:, i], time=time)
    r, number, euler_datas = phase2Rs(phase)
    np.savetxt(os.path.join(save_folder, 'r.csv'), r, delimiter=',')
    np.savetxt(os.path.join(save_folder, 'phase.csv'), phase, delimiter=',')
    r_plt(time, r, 'r.pdf')
    return r, phase


if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    # データのファイルの入っているフォルダに移動
    peak_folder = 'tmp/peak.csv'
    # ファイルを指定
    save_folder = 'tmp/R_result'
    dT = 60  # データの時間間隔を指定(分)
    n = 180  # どれだけの回数取ったか（20分おきで24時間なら72）

    peak2r(peak_folder, save_folder, dT=dT, n=n)
    # この関数で全部実行
