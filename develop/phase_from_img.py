# -*- coding: utf-8 -*-
"""A function that analyzes a time series image group for each pixel."""

import os
import sys

import image_analysis as im

import numpy as np

import pandas as pd

import peak_analysis as pa


def make_theta_imgs(imgs, avg=3, dt=60, p_range=13, f_avg=1, f_range=9, offset=0):
    """A function that obtains the peak for each pixel from the time series image by quadratic function fitting."""
    if avg % 2 - 1:
        sys.exit('avgは移動平均を取るデータ数です．奇数で指定してください．\n 他は前後 *_rangeに対して行っています')
    time = np.arange(imgs.shape[0], dtype=np.float64) * dt / 60 + offset  # 時間データを作成．
    use_xy = np.where(np.sum(imgs, axis=0) != 0)  # データの存在する場所のインデックスをとってくる．
    # 解析
    peak_a = pa.phase_analysis(imgs[:, use_xy[0], use_xy[1]], avg=avg, p_range=p_range, f_avg=f_avg, f_range=f_range, time=time)
    ###############################
    # 出力
    ###############################
    index = ["x", "y"]
    index.extend(range(peak_a[0].shape[0]))
    p_time = pd.DataFrame(np.vstack((use_xy, peak_a[0])), index=index)
    r2 = pd.DataFrame(np.vstack((use_xy, peak_a[3])), index=index)
    theta_imgs = np.empty(imgs.shape, dtype=np.float64)
    theta_imgs[:] = np.nan
    tau_imgs = np.copy(theta_imgs)
    peak_a[2][np.isnan(peak_a[2])] = -1
    peak_a[6][np.isnan(peak_a[6])] = -1
    theta_imgs[:, use_xy[0], use_xy[1]] = peak_a[2]
    tau_imgs[:, use_xy[0], use_xy[1]] = peak_a[6]
    return theta_imgs, tau_imgs, r2, peak_a[4], peak_a[5], p_time


def img_pixel_theta(folder, avg=3, mesh=1, dt=60, offset=0, p_range=12, f_avg=1, f_range=5, save=False, make_color=[22, 28], pdf=False, xlsx=False):
    """ピクセルごとに二次関数フィッティングにより解析する．
    
    位相・周期・Peak時刻，推定の精度 を出力する．
    
    Args:
        folder: 画像群が入ったフォルダを指定する．
        avg: ピーク位置を推定するときに使う移動平均 (default: {3})
        mesh: 解析前にメッシュ可する範囲 (default: {1} しない)
        dt: Minite (default: {60})
        offset: hour (default: {0})
        p_range: 前後それぞれp_rangeよりも値が高い点をピークとみなす (default: {12})
        f_avg: [description] (default: {1})
        f_range: 前後それぞれf_rangeのデータを用いて推定をする (default: {5})
        save: 保存先のフォルダ名．Trueなら自動で名付け． (default: {False})
        make_color: 周期を色付けする範囲 (default: {[22, 28]})
        pdf: PDFを保存するか否か (default: {False})
        xlsx: エクセルを保存するか否か (default: {False})
    
    Returns:
        保存のみの関数．返り値は持たさない(0)．
    """
    # 上の全部まとめたった！！
    if mesh == 1:
        imgs = im.read_imgs(folder)
    else:
        imgs = im.mesh_img(folder, mesh)
    # 生物発光の画像群から位相を求める
    peak_a = make_theta_imgs(imgs, avg=avg, dt=dt, p_range=p_range, f_avg=f_avg, f_range=f_range, offset=offset)
    # 位相のデータは醜いので，カラーのデータを返す．
    tau_frond = peak_a[1] == -1
    color_theta = im.make_colors(peak_a[0], grey=-1)
    imgs_tau = (peak_a[1] - make_color[0]) / (make_color[1] - make_color[0]) * 0.8
    nan = ~np.isnan(imgs_tau)
    imgs_tau[nan][imgs_tau[nan] > 0.8] = 0.8
    imgs_tau[nan][imgs_tau[nan] < 0] = 0
    imgs_tau[tau_frond] = -1
    color_tau = im.make_colors(imgs_tau, grey=-1)
    if save is not False:
        # color 画像の保存
        if save is True:
            save = os.path.split(folder)[0]
            save = os.path.join(save, '_'.join(['tau_mesh-' + str(mesh), 'avg-' + str(avg), 'prange-' + str(p_range), 'frange-' + str(f_range)]))
        im.save_imgs(os.path.join(save, 'theta'), color_theta)
        im.save_imgs(os.path.join(save, 'tau'), color_tau)
        # phaseを0−1で表したものをcsvファイルに
        np.save(os.path.join(save, 'theta.npy'), peak_a[0])
        np.save(os.path.join(save, 'tau.npy'), imgs_tau)
        # if pdf is not False:
        #     x = np.linalg.norm(np.asarray(
        #         peak_a[5]) - 80, axis=0).repeat(np.asarray(peak_a[2]).shape[0])
        #     y = np.asarray(peak_a[2]).reshape(len(x), order='F')
        #     x, y = x[~np.isnan(y)], y[~np.isnan(y)]
        #     max_x, max_y = 45, 1
        #     make_hst_fig(save_file=pdf, x=x, y=y, max_x=max_x,
        #                  max_y=max_y, max_hist=500, bin_hist=100)
        if xlsx is not False:
            writer = pd.ExcelWriter(os.path.join(save, "peak_list.xlsx"))
            peak_a[2].to_excel(writer, sheet_name='r2', index=True, header=False)  # 保存
            peak_a[5].to_excel(writer, sheet_name='peak_time', index=False, header=True)
            writer.save()
    return 0

if __name__ == "__main__":
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
    # カレントディレクトリの変更．
    #########################
    # パラメータ
    #########################
    folder = os.path.join('00data', '170613-LD2LL-ito-MVX', 'frond_180730', 'label-001_239-188_n214', 'small_moved_mask_frond_lum')
    save = os.path.join('_181120', 'test')

    img_pixel_theta(folder, avg=3, mesh=1, dt=60, offset=0, p_range=12, f_avg=1, f_range=5, save=save, make_color=[22, 28], xlsx=True)
