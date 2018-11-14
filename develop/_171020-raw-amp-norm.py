# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import image_analysis as im
import peak_analysis as pa


def img2only_data(imgs):
    # フロンドのあるピクセルだけにする．
    x_all = np.array(list(np.arange(imgs.shape[2]))*imgs.shape[1])
    y_all = np.repeat(np.arange(imgs.shape[1]), imgs.shape[2])
    imgs = imgs.reshape((imgs.shape[0], -1))
    frond_pixel = np.any(imgs != 0, axis=0)
    datas, x, y = imgs[:, frond_pixel], x_all[frond_pixel], y_all[frond_pixel]
    return datas, x, y


def cut_time_data(datas, state, s_point, e_point):
    # 全ての時間でデータがあるフロンドだけにする．
    cut_time = datas[s_point:e_point+1]
    use_data = np.all(cut_time != 0, axis=0)
    if state.ndim == 1:
        out_data, out_state = cut_time[:, use_data], state[use_data]
    else:
        out_data, out_state = cut_time[:, use_data], state[:, use_data]
    return out_data, out_state


def move_avg_long(data, w):  # wはデータ数
    if w % 2 == 1:
        w2 = int(w/2)
        w3 = w2+1
    else:
        w2 = w3 = int(w/2)
    move_data = np.empty_like(data).astype(np.float64)
    move_data[:w3] = np.average(data[:w], axis=0)
    move_data[-w2:] = np.average(data[-w:], axis=0)
    for i in range(w):
        move_data[w2:-w3] = move_data[w2:-w3] + data[i:-w+i]
    move_data[w2:-w3] = move_data[w2:-w3] / w
    return move_data


def move_sd_long(data, w):  # wはデータ数
    w2 = int(w/2)
    move_sd = np.empty_like(data).astype(np.float64)
    move_sd[:w2] = np.std(data[:w], axis=0, ddof=1)
    move_sd[-w2-1:] = np.std(data[-w:], axis=0, ddof=1)
    for i in range(data.shape[0]-w):
        move_sd[w2+i] = np.std(data[i:w+i], axis=0, ddof=1)
    return move_sd


def data2pdf(data, plot_data, dT=60, save_file='', coler='r', size=5, marker='.', linewidths=0):
    DT = dT/60
    time = np.arange(0, data.shape[0]*DT, DT, dtype=np.float64)
    pp = PdfPages(os.path.join(save_folder, 'raw.pdf'))
    for i in range(data.shape[1]):
        time_i = time[np.nonzero(plot_data[:, i])]
        data_i = data[np.nonzero(plot_data[:, i]), i]
        plt.subplot(5, 4, np.mod(i, 20)+1)
        plt.scatter(time_i, data_i[0], s=size, c=color, marker='.', linewidths=linewidths)
        plt.title(i)
        # plt.xlabel('time(h)')
        plt.xticks(np.arange(0, np.ceil(data.shape[0]*DT/24)*24, 24))  # メモリ
        plt.xlim([0, np.ceil(data.shape[0]*DT/24)*24])  # 範囲
        # y1軸の有効小数点を1桁にする
        # plt.gca().get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%.1f'))
        if np.mod(i, 20) == 19:
            print(i/20, data.shape[1])
            plt.rcParams['font.size'] = 6
            plt.tight_layout()  # レイアウト崩れを自動で直してもらう
            pp.savefig(dpi=10, transparent=True)
            plt.close()
    if np.mod(data.shape[1], 20) != 19:
        plt.rcParams['font.size'] = 6
        plt.tight_layout()
        pp.savefig(dpi=10, transparent=True)
        pp.close()
    return 0


def AmpNorm(data, w, dT=60, name=False, save_folder=False, pdf=False):  # wは正規化の範囲
    DT = dT/60
    move_data, sd = move_avg_long(data, w=w), move_sd_long(data, w=w)
    norm_data = np.zeros_like(move_data)
    norm_data[data.nonzero()] = data[data.nonzero()] - move_data[data.nonzero()]
    norm_data[data.nonzero()] = norm_data[data.nonzero()] / sd[data.nonzero()]
    sd_data = sd
    sd_data[data == 0] = np.NaN
    if save_folder is not False:
        np.savetxt(os.path.join(save_folder, 'norm_data_float.csv'), norm_data, delimiter='\t')
        np.savetxt(os.path.join(save_folder, 'name.csv'), name, delimiter='\t')
    if pdf is not False:
        time = np.arange(0, data.shape[0]*DT, DT, dtype=np.float64)
        pp = PdfPages(os.path.join(save_folder, 'norm_data.pdf'))
        for i in range(data.shape[1]):
            time_i = time[np.nonzero(norm_data[:, i])]
            data_i = norm_data[np.nonzero(norm_data[:, i]), i]
            plt.subplot(5, 4, np.mod(i, 20)+1)
            plt.scatter(time_i, data_i[0], s=5, c='r', marker='.', linewidths=0)
            plt.title(i)
            # plt.xlabel('time(h)')
            plt.xticks(np.arange(0, np.ceil(move_data.shape[0]*DT/24)*24, 24))  # メモリ
            plt.xlim([0, np.ceil(move_data.shape[0]*DT/24)*24])  # 範囲
            # y1軸の有効小数点を1桁にする
            plt.gca().get_yaxis().set_major_formatter(ticker.FormatStrFormatter('%.1f'))
            if np.mod(i, 20) == 19:
                print(i/20, data.shape[1])
                plt.rcParams['font.size'] = 6
                plt.tight_layout()  # レイアウト崩れを自動で直してもらう
                pp.savefig(dpi=10)
                plt.close()
        if np.mod(move_data.shape[1], 20) != 19:
            plt.rcParams['font.size'] = 6
            plt.tight_layout()
            pp.savefig(dpi=10)
        pp.close()
    return norm_data


def raw_save(data, dT=60, name=False, save_folder=False, pdf=False):  # wは正規化の範囲
    DT = dT/60
    if save_folder is not False:
        np.savetxt(os.path.join(save_folder, 'raw_data.csv'), data, delimiter='\t')
        np.savetxt(os.path.join(save_folder, 'raw_name.csv'), name, delimiter='\t')
    if pdf is not False:
        save_file = os.path.join(save_folder, 'raw.pdf')
        data2pdf(data, dT=dT, plot_data=data, save_file=save_file, coler='r', size=5, marker='.', linewidths=0)
    return data


if __name__ == '__main__':
    os.chdir('/hdd1/Users/kenya/Labo/keisan/python/00data')
    # days = (['./170215-LL2LL-MVX'])
    days = (['./170613-LD2LL-ito-MVX'])  # , './170829-LL2LL-ito-MVX']

    dT = 60

    # 中心のx，y座標．
    C_X = 80
    C_Y = 80

    # 頭悪かりし頃に書いたので，列名とかは別ファイルに分けてある．
    # つまり，全てのセルにデータが入っている状態．

    # 解析データのフォルダ
    for day in days:  # 僕が色々一括でやりたかっただけ．不要．
        frond_folder = os.path.join(day, 'frond')
        for i in sorted(glob.glob(os.path.join(frond_folder, '*'))):  # 不要
            print(i)
            ################### ここから．ピクセル毎にする場合 ########
            folder_img_small = 'small_moved_mask_frond_lum'  # 画像のファイル
            img = im.read_imgs(os.path.join(i, folder_img_small))
            # とりあえず座標の設定と画像の整形
            datas, x, y = img2only_data(img)
            # ここで距離の計算
            pixel_linalg = np.linalg.norm((np.array([x-C_X, y-C_Y])), axis=0)

            name = np.array([x, y, pixel_linalg])
            name = np.array(name)
            # ここから，エクセルデータを取り込むとき．
            # data = pd.read_csv('data/src/sample.csv', header=0, index_col=None, sep=',')
            # datas = data.values
            # name = False
            # 保存するフォルダを指定して．
            save_folder = os.path.join('result' + day.split('/')[1], os.path.split(i)[1])
            if os.path.exists(save_folder) == False:
                os.makedirs(save_folder)
            ######################### 名前をつける #########################

            data_AmpNorm = AmpNorm(datas, w=24, dT=dT, name=name, save_folder=save_folder, pdf=False)  # 振幅を標準化する．
            raw_save(datas, dT=dT, save_folder=save_folder, pdf=False, name=name)  # そのまま
            # PDFは，ピクセルごとの解析の場合はTrueにしないこと！！
