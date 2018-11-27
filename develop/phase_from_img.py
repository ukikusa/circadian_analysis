# -*- coding: utf-8 -*-
"""A function that analyzes a time series image group for each pixel."""

import os
import sys

import image_analysis as im
from make_figure import make_hst_fig
# import matplotlib as mpl
# mpl.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import peak_analysis as pa
from PIL import Image  # Pillowの方を入れる．PILとは共存しない


def make_theta_imgs(imgs, mask_img=False, avg=3, dt=60, p_range=13, f_avg=1, f_range=9, offset=0, r2_cut=0.5):
    """画像データから二次関数フィティングで位相を出す.

    Args:
        imgs: 投げる画像
        avg: [description] (default: {3})
        dt: [description] (default: {60})
        p_range: [description] (default: {13})
        f_avg: [description] (default: {1})
        f_range: [description] (default: {9})
        offset: [description] (default: {0})

    Returns:
        位相(0-1)画像, 周期画像, r2値, peak_a[4], peak_a[5], ピーク時間
        [type]
    """
    if avg % 2 - 1:
        sys.exit('avgは移動平均を取るデータ数です．奇数で指定してください．\n 他は前後 *_rangeに対して行っています')
    time = np.arange(imgs.shape[0], dtype=np.float64) * dt / 60 + offset  # 時間データを作成．
    use_xy = np.where(np.sum(imgs, axis=0) != 0)  # データの存在する場所のインデックスをとってくる．
    # 解析
    peak_a = pa.phase_analysis(imgs[:, use_xy[0], use_xy[1]], avg=avg, p_range=p_range, f_avg=f_avg, f_range=f_range, time=time, r2_cut=r2_cut)
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
    if mask_img is False:
        peak_a[2][np.isnan(peak_a[2])] = -1
        peak_a[6][np.isnan(peak_a[6])] = -1
    else:
        peak_a[2][np.isnan(peak_a[2]) * mask_img != 0] = -1
        peak_a[6][np.isnan(peak_a[6]) * mask_img != 0] = -1
    theta_imgs[:, use_xy[0], use_xy[1]] = peak_a[2]
    tau_imgs[:, use_xy[0], use_xy[1]] = peak_a[6]
    return theta_imgs, tau_imgs, r2, peak_a[4], peak_a[5], p_time, use_xy


def make_peak_img(theta_imgs, mask_imgs=False, idx_t=[24, 24 * 5 + 1]):
    """位相のArrayからピーク画像のArrayを出力

    Args:
        theta_imgs: 位相のArray
        mask_imgs: 背景画像のArray (default: {False})
        idx_t: 出力するIndex (default: {[24, 24 * 5 + 1]})

    Returns:
        ピーク255，背景50，その他0のArray
    """
    theta_imgs = theta_imgs[idx_t[0]: idx_t[1]]
    # 必要な部分だけ切り出し
    diff = np.diff(theta_imgs, n=1, axis=0)  # どの画像とどの画像の間でピークが来ているかを計算
    peak_img = np.zeros_like(diff, dtype=np.uint8)
    if mask_imgs is not False:
        # フロンドのある場所を描写
        mask_imgs = mask_imgs[idx_t[0]: idx_t[1]]
        peak_img[mask_imgs[:-1] != 0] = 50
    else:
        peak_img[~np.isnan(theta_imgs[:-1])] = 50
    # peakの来ている場所を描写
    peak_img[diff < 0] = 255
    peak_img[diff >= 1] = 255
    return peak_img


def peak_img_list(peak_img, per_ber=10, m=5, fold=24):
    """Peak画像のArrayにBerつけて画像保存できる形で出力．

    Args:
        peak_img: ピーク画像のArray(2次元)
        per_ber: ピーク割合のバーの太さ (default: {10})
        m: 折返しの白線 (default: {5})
        fold: 画像を折り返すか (default: {24})

    Returns:
        Peak画像
    """
    if per_ber == 0 or False:
        return peak_img
    n, xsize, ysize = peak_img.shape[0:3]  # 画像サイズ
    peak = np.sum(peak_img == 255, axis=(1, 2))
    frond = np.sum(peak_img != 0, axis=(1, 2))
    print(xsize)
    ber_len = xsize - 14  # berの長さ
    ber = np.tile(np.arange(ber_len), (n, per_ber, 1))  # ber の下地
    idx = (ber.T <= peak.astype(np.float64) * ber_len / frond).T  # 塗りつぶす場所
    ber[idx] = 255
    ber[~idx] = 50
    if m == 0:
        peak_img[:, - per_ber:, 7: xsize - 7] = ber  # berをマージ
    else:
        peak_img[:, -m - per_ber:-m, 7: xsize - 7] = ber  # berをマージ

    # ピークと判定されたピクセル数のバーを出す．
    peak_img = np.reshape(peak_img.transpose(1, 2, 0), (ysize, -1), 'F')  # 画像を横に並べる．
    peak_img[:m] = 255
    if fold is not False:
        peak_img = np.vstack(np.hsplit(peak_img, int(n / fold)))
    return peak_img


def img_analysis_pdf(save_file, tau, distance_center=True, dt=60):
    """作図．諸々の解析の．"""
    pp = PdfPages(save_file)
    # fig = plt.figure(1, figsize=(6, 4), dpi=100)
    if distance_center is True:
        distance_center = (np.array(tau.shape[1:]).astype(np.float64) * 0.5).astype(np.uint8)
    time = np.arange(0, tau.shape[0], 24 * 60 / dt).astype(np.uint8)
    tau = tau[time]
    tau_nan = np.logical_or(np.isnan(tau), tau <= 14)
    tau_nan = np.logical_or(tau_nan, tau >= 30)
    fig_n = time.shape[0]
    for i in range(1, fig_n):
        if np.sum(~tau_nan):
            pass
        tau_idx = np.array(np.where(~tau_nan[i]))
        distance = np.linalg.norm((tau_idx.T - distance_center), axis=1)
        title = str(time[i]) + '(h) center[ ' + str(distance_center[0]) + ', ' + str(distance_center[1]) + ']'
        print(distance.shape, tau[i][~tau_nan[i]].shape)
        make_hst_fig(save_file=save_file, x=distance, y=tau[i][~tau_nan[i]], min_x=0, max_x=None, min_y=20, max_y=30, max_hist_x=200, max_hist_y=200, bin_hist_x=200, bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='distance(pixcel)', ylabel='period(h)', pdfpages=pp, box=1, per=True, title=title)
    pp.close()
    return 0


def img_pixel_theta(folder, mask_folder=False, avg=3, mesh=1, dt=60, offset=0, p_range=12, f_avg=1, f_range=5, save=False, make_color=[22, 28], pdf=False, xlsx=False, distance_center=True, r2_cut=0.5):
    """ピクセルごとに二次関数フィッティングにより解析する．

    位相・周期・Peak時刻，推定の精度 を出力する．

    Args:
        folder: 画像群が入ったフォルダを指定する．
        mask_folder: 背景が0になっている画像群を指定．
        mesh: 解析前にメッシュ化する範囲 (default: {1} しない)
        dt: Minite (default: {60})
        offset: hour (default: {0})
        p_range: 前後それぞれp_rangeよりも値が高い点をピークとみなす (default: {12})
        f_avg: fiting前の移動平均 (default: {1})
        f_range: 前後それぞれf_rangeのデータを用いて推定をする (default: {5})
        save: 保存先のフォルダ名．Trueなら自動で名付け． (default: {False})
        make_color: 周期を色付けする範囲 (default: {[22, 28]})
        pdf: PDFを保存するか否か (default: {False})
        xlsx: エクセルを保存するか否か (default: {False})
        distance_center: 作図の際どこからの距離を取るか (default: {Ture} center)

    Returns:
        保存のみの関数．返り値は持たさない(0)．
    """
    # 上の全部まとめたった！！
    if mesh == 1:
        imgs = im.read_imgs(folder)
        if mask_folder is not False:
            mask = im.read_imgs(mask_folder)
        else:
            mask = False
    else:
        imgs = im.mesh_img(folder, mesh)
        if mask_folder is not False:
            mask = im.mesh_img(mask_folder, mesh)
        else:
            mask = False
    ##################################
    # 生物発光の画像群から位相を求める
    ##################################
    peak_a = make_theta_imgs(imgs, avg=avg, dt=dt, p_range=p_range, f_avg=f_avg, f_range=f_range, offset=offset, r2_cut=r2_cut)
    # 位相のデータは醜いので，カラーのデータを返す．

    color_theta = im.make_colors(peak_a[0], grey=-1)
    ###################################
    # 保存用に周期を整形
    ###################################
    tau_frond = peak_a[1] == -1
    imgs_tau = (peak_a[1] - make_color[0]) / (make_color[1] - make_color[0]) * 0.8
    nan = ~np.isnan(imgs_tau)
    imgs_tau[nan][imgs_tau[nan] > 0.8] = 0.8
    imgs_tau[nan][imgs_tau[nan] < 0] = 0
    imgs_tau[tau_frond] = -1
    color_tau = im.make_colors(imgs_tau, grey=-1)
    ####################################
    # Peakの画像の作成
    ####################################
    fold = int(60 / dt) * 24
    p_img = make_peak_img(peak_a[0], mask_imgs=mask, idx_t=[0, int(peak_a[0].shape[0] / fold) * fold + 1])
    p_img = peak_img_list(p_img, per_ber=10, m=0, fold=fold)
    if save is not False:
        # color 画像の保存
        if save is True:
            save = os.path.split(folder)[0]
            save = os.path.join(save, '_'.join(['tau_mesh-' + str(mesh), 'avg-' + str(avg), 'prange-' + str(p_range), 'frange-' + str(f_range)]))
        im.save_imgs(os.path.join(save, 'theta'), color_theta)
        im.save_imgs(os.path.join(save, 'tau'), color_tau)
        Image.fromarray(p_img).save(os.path.join(save, 'peak_img.png'), compress_level=0)
        # phaseを0−1で表したものをcsvファイルに
        np.save(os.path.join(save, 'theta.npy'), peak_a[0])
        np.save(os.path.join(save, 'tau.npy'), imgs_tau)
        if pdf is not False:
            if pdf is True:
                pdf = os.path.join(save, 'analysis.pdf')
            else:
                pdf = os.path.join(save, pdf)
            img_analysis_pdf(save_file=pdf, tau=peak_a[1], distance_center=distance_center, dt=dt)
        if xlsx is not False:
            writer = pd.ExcelWriter(os.path.join(save, "peak_list.xlsx"))
            peak_a[2].T.to_excel(writer, sheet_name='r2', index=False, header=True)  # 保存
            peak_a[5].T.to_excel(writer, sheet_name='peak_time', index=False, header=True)
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
