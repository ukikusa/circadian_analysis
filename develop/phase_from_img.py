# -*- coding: utf-8 -*-
"""Phase from image and fig."""

import glob
import os
import sys

import image_analysis as im

import numpy as np

import pandas as pd

import peak_analysis as pa
# from _171020_make_period_img import make_period_imgs


def make_phase_imgs(imgs, avg, dt=60, p_range=13, f_range=9, csv_save=False, offset=0):
    """A function that obtains the peak for each pixel from the time series image by quadratic function fitting."""
    if avg % 2 - 1:
        sys.exit('avgは移動平均を取るデータ数です．奇数で指定してください．\n 他は前後 *_rangeに対して行っています')
    phase_imgs = np.empty(imgs.shape, dtype=np.float64)  # 位相の出力画像を設定．
    phase_imgs[:] = np.nan
    period_imgs = np.copy(phase_imgs)  # 周期の出力画像を設定．
    time = np.arange(imgs.shape[0], dtype=np.float64) * dt / 60 + offset  # 時間データを作成．
    use_xy = np.where(np.sum(imgs, axis=0) != 0)  # データの存在する場所のインデックスをとってくる．
    label = use_xy[0].astype(str).astype(object) + ',' + use_xy[1].astype(str).astype(object)
    print(imgs.shape)
    print(imgs[:, use_xy[0], use_xy[1]].shape)
    # ここでピーク抽出やグラフ作成を終わらす
    peak_a = pa.phase_analysis(imgs[:, use_xy[0], use_xy[1]], avg=avg, dt=dt, p_range=p_range, f_range=f_range, time=time)
    peak_time = pd.DataFrame(peak_a[0], columns=label)
    r2 = pd.DataFrame(peak_a[3], columns=label)
    phase_imgs[:, use_xy[0], use_xy[1]] = peak_a[2]
    period_imgs[:, use_xy[0], use_xy[1]] = peak_a[6]
    return phase_imgs, period_imgs, r2, peak_a[4], peak_a[5], use_xy, peak_time


def img_to_mesh_phase(folder, avg, mesh=1, dt=60, offset=0, p_range=12, f_range=5, save_imgs=False, make_color=[22, 28], save_pdf=False, save_xlsx=False):
    # 上の全部まとめたった！！
    if mesh == 1:
        imgs = im.read_imgs(folder)
    else:
        imgs = im.mesh_img(folder, mesh)
    # 生物発光の画像群から位相を求める
    peak_a = make_phase_imgs(imgs, avg=avg, dt=dt, p_range=p_range, f_range=f_range, offset=offset)
    imgs_phase = peak_a[0]
    # 位相のデータは醜いので，カラーのデータを返す．
    color_phase = im.make_colors(imgs_phase)
    imgs_period = (peak_a[1] - make_color[0]) / (make_color[1] - make_color[0]) * 0.8
    nan = ~np.isnan(imgs_period)
    imgs_period[nan][imgs_period[nan] > 0.8] = 0.8
    imgs_period[nan][imgs_period[nan] < 0] = 0
    color_period = im.make_colors(imgs_period)
    if save_imgs is not False:
        # color 画像の保存
        im.save_imgs(os.path.join(save_folder, 'phase' + '_mesh' + str(mesh) + '_avg' + str(avg)), color_phase)
        im.save_imgs(os.path.join(save_folder, 'period' + '_mesh' + str(mesh) + '_avg' + str(avg)), color_period)
        # phaseを0−1で表したものをcsvファイルに
        np.save(save_folder + '/small_phase' + '_mesh' + str(mesh) + '_avg' + str(avg) + '.npy', imgs_phase)
        np.save(os.path.join(save_folder, 'small_period' + '_mesh' + str(mesh) + '_avg' + str(avg) + '.npy'), imgs_period)
    if save_pdf is not False:
        x = np.linalg.norm(np.asarray(
            peak_a[5]) - 80, axis=0).repeat(np.asarray(peak_a[2]).shape[0])
        y = np.asarray(peak_a[2]).reshape(len(x), order='F')
        x, y = x[~np.isnan(y)], y[~np.isnan(y)]
        max_x, max_y = 45, 1
        make_hst_fig(save_file=save_pdf, x=x, y=y, max_x=max_x,
                     max_y=max_y, max_hist=500, bin_hist=100)
    return color_phase, imgs_phase, peak_a[2]


# dataはdata,avgで移動平均．前後p_range分より高い点を暫定ピークに．その時の移動平均がpeak_avg．fit_rangeでフィッティング

if __name__ == "__main__":
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan',
                          'python', '00data'))  # カレントディレクトリの移動．ググって．ないしは，下のフォルダ指定で絶対パス指定をする．

    # 解析データのフォルダ
    data_folder = os.path.join(
        '.', 'small_moved_mask_frond_lum')  # 助長かも．データのあるフォルダを指定して．
    save_folder = '.'
    out = img_to_mesh_phase(data_folder, avg=3, mesh=1, dt=60, peak_avg=3, p_range=12, f_range=5, save_folder=save_folder, pdf_save=os.path.join(save_folder, 'tmp.pdf'))

    sys.exit('正常に終了')
    ################
    # roop回したければ以下の通り
    ####################
    days = ['./170215-LL2LL-MVX',
            './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX']
    for day in days:
        frond_folder = day + '/frond'
        for i in sorted(glob.glob(frond_folder + '/*')):
            print(i)
            # 解析データのフォルダ
            data_folder = i + '/small_moved_mask_frond_lum/'
            save_folder = i
            out = img_to_mesh_phase(data_folder, avg=3, mesh=1, dt=60, peak_avg=3, p_range=13, f_range=5, save_folder=save_folder, pdf_save=os.path.join(save_folder, 'tmp.pdf'))
