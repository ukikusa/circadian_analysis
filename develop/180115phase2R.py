# -*- coding: utf-8 -*-

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa


def phase2R(phase_data):
    # phaseのデータを投げたら同期率Rと位相のリストを返す関数.2次元配列のみ対応．
    # オイラー展開（でいいのかな）するよ
    euler_data = np.exp(1j * phase_data[np.logical_xor(phase_data != -1, np.isnan(phase_data))] * 2 * np.pi)
    # 余裕がアレば，位相の画像返してあげてもいいね．
    R = np.sum(euler_data) / int(np.size(euler_data))
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


def frond_plt(r, number, area, lum_sum, pdf_save):
    fig1 = plt.subplot(2, 1, 1)
    fig1.set_title(pdf_save + ' all_bio')
    x_axis = np.arange(0, len(r) * 1, 1)
    xticks = np.arange(0, np.ceil(len(r) / 24) * 24, 24)
    # x軸の作成．ここは時間にしないと
    fig1.plot(x_axis, lum_sum, color='r')
    fig1.set_xlabel('time(h)')
    fig1.set_xticks(xticks)
    plt.xlim([0, np.ceil(len(r) / 24) * 24])  # x軸の条件
    plt.ylim(ymin=0)  # y軸の条件
    # 次のグラフ作成
    ax1 = plt.subplot(2, 1, 2)  # プロットわけ，プロット位置
    ax1.set_title(pdf_save + ' R')
    ax1.plot(x_axis, r, color='r')
    # Rのグラフを作成
    ax1.set_ylim([0, 1])
    # y軸を0から1に
    ax1.set_xlabel('time')
    ax1.set_ylabel('R')
    plt.xticks(xticks)
    plt.xlim([0, np.ceil(len(r) / 24) * 24])  # x軸の条件
    plt.ylim(ymin=0)  # y軸の条件    plt.xlim([0, np.ceil(len(r)/24)*24])
    # ラベル付
    ax2 = ax1.twinx()
    ax2.plot(x_axis, number)
    ax2.plot(x_axis, area, color='y')
    # 面積をかぶせる
    ax2.set_ylabel('area(pixel)')
    ax1.set_xticks(xticks)

    plt.rcParams['font.size'] = 10
    # レイアウト崩れを自動で直してもらう
    plt.tight_layout()
    plt.savefig(pdf_save + '.pdf')
    plt.close()
    return r, number


def frond_r_2fig(r, number, col, pdf_save, dT=60, offset=0):
    # dataの取り込み,タブ区切りなら拡張子をtsv．　delimiter='\t'でできる気がする試してない
    # 時間の定義
    time = np.arange(data.shape[0], dtype=float) * dT / 60 + offset
    # if avg is not False:
    #    time_move = time[int(avg / 2): data.shape[0] - int(avg / 2)]
    # else:
    #     time_move = time
    # peakや時間を出力する箱を作る．
    plt.rcParams['font.size'] = 7
    if os.path.exists(pdf_save) is False:
        os.makedirs(pdf_save)
    # plt.figure(figsize=(8.27, 11.69), dpi=100) # 余白なしA4
    plt.figure(figsize=(6.5, 9), dpi=100)  # A4 余白を持って
    for i in range(r.shape[0]):
        # pdf一ページに対して，配置するグラフの数．配置するグラフの場所を指定．
        plt.subplot(7, 4, np.mod(i, 28) + 1)
        # peakを推定する．func_valueで計算．
        # if avg is not False:
        #     data_move = pa.moving_avg(data[:, i], avg=avg)
        # else:
        #     data_move = data[:, i]
        # x軸の調整
        plt.plot(time, r[i], color='r', lw=1, label=col[i])
        plt.plot(time, number[i], color='b', lw=1)
        plt.ylim([0, 5000])
        # plt.ylim(bottom=0)
        plt.tick_params(labelbottom='off', labelleft='off')
        plt.tick_params(right="on", top="off")
        # plt.xlabel('time'
        plt.xlim([0, 192])
        plt.xticks(np.arange(0, 192, 24))
        plt.legend(loc='best', frameon=1, framealpha=1, handlelength=False, markerscale=False, fontsize=7, edgecolor='w')
        plt.vlines(np.arange(0, 192, 24), 0, 5000, colors="k", linestyle="dotted", label="", lw=0.5)

        if np.mod(i, 28) == 27:
            # レイアウト崩れを自動で直してもらう
            plt.tight_layout()
            # 保存
            plt.savefig(os.path.join(pdf_save, str(i) + ".pdf"))
            plt.close()
            plt.figure(figsize=(6.5, 9), dpi=100)
    if np.mod(i, 28) != 27:
        plt.rcParams['font.size'] = 7
        plt.tight_layout()
        plt.savefig(os.path.join(pdf_save, str(i) + ".pdf"))
        plt.close()
    return 0


if __name__ == '__main__':
    # グラフ化
    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    days = ['./170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX', './170215-LL2LL-MVX']
    for day in days:
        frond_folder = day + '/frond'
        data_file = '/small_phase_mesh1_avg3.npy'
        r, number, col = [], [], []
        dT = 60
        for i in sorted(glob.glob(frond_folder + '/*')):
            print(i)
            # 解析データのフォルダ
            data = np.load(i + data_file)
            save_folder = day + '/result/small_R_avg3/' + (data_file.lstrip('/')).rstrip('.npy') + str('_') + i.split('/')[-1]
            if os.path.exists(day + '/result/small_R_avg3') is False:
                os.makedirs(day + '/result/small_R_avg3')
            r_tmp, number_tmp, a = phase2Rs(data)
            r.append(r_tmp)
            number.append(number_tmp)
            col.append(i.split('/')[-1][7:9])
        r = np.array(r) * 5000
        print(r.shape)
        number = np.array(number, dtype=np.float64)
        r[number < 1000] = np.nan
        number[number < 1000] = np.nan
        pdf_save = os.path.join(day)
        frond_r_2fig(r, number, col, pdf_save, dT, offset=0)
    # np.savetxt('/hdd1/kenya/Labo/keisan/python/2017_06_13LDtoLL/phase.csv', result, fmt = '%.4f%+.4fj, '*512, delimiter=",")
        # pdf一ページに対して，配置するグラフの数．配置するグラフの場所を指定．
        # plt.subplot(4, 3, np.mod(i, 12)+1)
        # vplt.title(i)
        # x軸の調整
        # plt.xlabel('time')
        # plt.xticks(np.arange(time[0], time[-1], 24)-time[0])
        # if np.mod(i, 12) == 11:
        # 文字サイズを一括調整
        # plt.rcParams['font.size'] = 6
        # レイアウト崩れを自動で直してもらう
        # plt.tight_layout()
        # 保存
        # plt.savefig(file + '/' + str(i) + '.pdf')
        # plt.close()
