# utf-8

import numpy as np
import os
# import map
import cv2
import sys
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
from image_analysis import make_color


def make_frond_phase_imgs(imgs, label_imgs, avg, dT=60, peak_avg=3, p_range=6, fit_range=4, csv_save=False, pdf_save=False):
    if os.path.exists(pdf_save) is False:
        os.mkdir(pdf_save)
        print(pdf_save + 'フォルダを作成しました')
    label = np.unique(label_imgs[np.nonzero(label_imgs)])
    data = np.zeros((imgs.shape[0], label.shape[0]), dtype=np.float64)
    phase_imgs = np.ones(imgs.shape, dtype=np.float64) * -1
    for i in range(label_imgs.shape[0]):
        img = imgs[i]
        label_img = label_imgs[i]
        for j, k in enumerate(label):
            data[i, j] = np.average(img[label_img == k])
    phase_data = np.ones(data.shape, dtype=np.float64) * -1
    time = np.arange(imgs.shape[0], dtype=float) * dT / 60
    time_move = time[int(avg / 2): imgs.shape[0] - int(avg / 2)]
    # それぞれのデータに名前をつけるときように，一応データ名を取り込めるようにしておく．pandaを使った方がいいかも．
    # header = np.loadtxt('data.csv', dtype='U15', delimiter=',')
    # peakや時間を出力する箱を作る．
    peak_time = np.zeros((int(data.shape[0] / p_range), data.shape[1]), dtype=np.float64)
    peak_value = np.zeros_like(peak_time)
    func_value = np.zeros((int(data.shape[0] / p_range), data.shape[1], 3))
    # 全部にループ回すのは頭が悪いので削る
    # use_index = np.where(np.average(data, axis=0)>1500)[0]
    # それぞれの時系列に対して処理を回す．ここでピーク抽出やグラフ作成を終わらす．
    # print(use_index)
    for i in range(data.shape[1]):
        if pdf_save is not False:
            # pdf一ページに対して，配置するグラフの数．配置するグラフの場所を指定．
            plt.subplot(4, 3, np.mod(i, 12) + 1)
            # peakを推定する．func_valueで計算．
        peak_time_tmp, peak_value_tmp, func_value_tmp, peak_point, pcov = pa.peak_find(data[:, i], time=time_move, avg=avg, peak_avg=peak_avg, p_range=p_range, fit_range=fit_range, pdf_save=pdf_save)
        if pdf_save is not False:
            plt.title(i)
            # x軸の調整
            plt.xlabel('time')
            plt.xticks(np.arange(time[0], time[-1], 24) - time[0])
            if np.mod(i, 12) == 11:
                # 文字サイズを一括調整
                plt.rcParams['font.size'] = 6
                # レイアウト崩れを自動で直してもらう
                plt.tight_layout()
                # 保存
                plt.savefig(pdf_save + '/' + str(i) + '.pdf')
                plt.close()
            # peak_time.append(peak_time_tmp)
            # peak_value.append(peak_value_tmp)
        peak_time[0:len(peak_time_tmp), i] = peak_time_tmp
        phase_data[:, i] = phase_fit(peak_time_tmp, data.shape[0])
    else:
        if pdf_save is not False:
            plt.title(i)
            plt.xlabel('time')
            plt.xticks(np.arange(time[0], time[-1], 24) - time[0])
            plt.rcParams['font.size'] = 6
            plt.tight_layout()
            plt.savefig(pdf_save + '/' + str(i) + '.pdf')
            plt.close()
        # print(peak_time_tmp)
        peak_time[peak_time == 0] = np.nan
        if csv_save is not False:
            np.savetxt(csv_save, peak_time, delimiter=',')
            np.savetxt(csv_save + 'data.csv', data, delimiter=',')
        phase_data = np.array(phase_data, dtype=np.float)
    for i in range(data.shape[0]):
        for j, k in enumerate(label):
            phase_imgs[i, label_imgs[i] == k] = phase_data[i, j]
    return phase_imgs


def mesh_img(folder, mesh=5):
    # 画像のフォルダから画像を全部読み込んできて，全てメッシュ化してしまおう！
    img = im.read_imgs(folder)
    # 以下メッシュ化するよ！
    meshed = np.empty((img.shape[0], int(img.shape[1] / mesh), int(img.shape[2] / mesh)))
    for i in range(int(img.shape[1] / mesh)):
        for j in range(int(img.shape[2] / mesh)):
            meshed[::, i, j] = img[::, i * mesh:(i + 1) * mesh, j * mesh:(j + 1) * mesh].mean(axis=(1, 2))
    return meshed


def img_to_frond_phase(folder, label_folder, avg, mesh=3, dT=60, peak_avg=3, p_range=12, fit_range=5, save=True, csv_save=False, pdf_save=False):
    # 上の全部まとめたった！！
    # メッシュ化
    if mesh == 1:
        data = im.read_imgs(folder)
        label_img = im.read_imgs(label_folder)
    else:
        data = mesh_img(folder, mesh)
        label_img = im.read_imgs(label_folder)
    # 解析をする．
    imgs_phase = make_frond_phase_imgs(imgs=data, label_imgs=label_img, avg=avg, dT=dT, peak_avg=peak_avg, p_range=p_range, fit_range=fit_range, csv_save=csv_save, pdf_save=pdf_save)
    imgs_phase[np.logical_and(label_img != 0, imgs_phase == -1)] = -2
    color_phase = np.empty((imgs_phase.shape[0], imgs_phase.shape[1], imgs_phase.shape[2], 3))
    for i in range(imgs_phase.shape[0]):
        color_phase[i] = make_color(imgs_phase[i, ::, ::], glay=True)
    if save == True:
        # color 画像の保存
        # imgs_merge = mesh_img(day + '/frond_170627224290/', mesh=mesh)
        im.save_imgs(folder.rstrip('/') + '_mesh' + str(mesh) + '_avg' + str(avg) + 'phase_color_merged', color_phase)
        # phaseを0−1で表したものをcsvファイルに
        np.save(folder.rstrip('/') + '_mesh' + str(mesh) + '_avg' + str(avg) + 'phase_merged.npy', imgs_phase)
    return color_phase, imgs_phase


# dataはdata,avgで移動平均．前後p_range分より高い点を暫定ピークに．その時の移動平均がpeak_avg．fit_rangeでフィッティング

if __name__ == '__main__':
    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    # フロンド全体の発光合計のリズム解析
    data_file = os.path.join('edit_raw', 'lum_min_img')
    label_file = os.path.join('edit_raw', 'label_img')

    day = os.path.join('.', '170215-LL2LL-MVX')
    # dT = 60
    dT = 60 + 10 / 60
    # 解析データのフォルダ
    data_folder = os.path.join(day, data_file)  # 発光画像
    label_folder = os.path.join(day, label_file)  # ラベル(使わないところは0)
    # 出力先フォルダ
    color, imgs_phase = img_to_frond_phase(data_folder, label_folder, avg=3, mesh=1, dT=60, peak_avg=3, p_range=12, fit_range=5, csv_save=day + 'frond_peak_csv', pdf_save=day + '/pdf')
