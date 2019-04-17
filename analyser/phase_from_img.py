# -*- coding: utf-8 -*-
"""A function that analyzes a time series image group for each pixel."""

import glob
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import FFT_nlls
import image_analysis as im
from make_figure import make_hst_fig
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
import peak_analysis as pa
from PIL import Image  # Pillowの方を入れる．PILとは共存しない


def make_theta_imgs(imgs, mask_img=False, avg=3, dt=60, p_range=13, f_avg=1, f_range=9, offset=0, r2_cut=0.5, min_tau=16, max_tau=32, amp_r=24 * 3):
    """画像データから二次関数フィティングで位相を出す.

    Args:
        imgs: 投げる画像
        avg: [description] (default: {3})
        dt: [description] (default: {60})
        p_range: [description] (default: {13})
        f_avg: [description] (default: {1})
        f_range: [description] (default: {9})
        offset: [description] (default: {0})
        amp_r: [estimate ampritude range (h)] (default 72)

    Returns:
        位相(0-1)画像, 周期画像, r2値, peak_a[4], peak_a[5], ピーク時間
        [type]
    """
    if avg % 2 - 1:
        sys.exit('avgは移動平均を取るデータ数です．奇数で指定してください．\n 他は前後 *_rangeに対して行っています')
    time = np.arange(imgs.shape[0], dtype=np.float64) * dt / 60 + offset  # 時間データを作成．
    use_xy = np.where(np.sum(imgs, axis=0) != 0)  # データの存在する場所のインデックスをとってくる．
    # 解析
    peak_a = pa.phase_analysis(imgs[:, use_xy[0], use_xy[1]], avg=avg, p_range=p_range, f_avg=f_avg, f_range=f_range, time=time, r2_cut=r2_cut, min_tau=min_tau, max_tau=max_tau)
    cv, sd = pa.amp_analysis(imgs[:, use_xy[0], use_xy[1]], int(60 / dt * amp_r))
    ###############################
    # 出力
    ###############################
    index = ["x", "y"]
    index.extend(range(peak_a[0].shape[0]))
    p_time = pd.DataFrame(np.vstack((use_xy, peak_a[0])), index=index)
    r2 = pd.DataFrame(np.vstack((use_xy, peak_a[3])), index=index)
    theta_imgs = np.full_like(imgs, np.nan, dtype=np.float64)
    tau_imgs, cv_imgs, sd_imgs = np.copy(theta_imgs), np.copy(theta_imgs), np.copy(theta_imgs)
    # if mask_img is False:
    #     peak_a[2][np.isnan(peak_a[2])] = -1
    #     peak_a[6][np.isnan(peak_a[6])] = -1
    #     cv[np.isnan(cv)] = -1
    #     sd[np.isnan(sd)] = -1
    theta_imgs[:, use_xy[0], use_xy[1]] = peak_a[2]
    tau_imgs[:, use_xy[0], use_xy[1]] = peak_a[6]
    cv_imgs[:, use_xy[0], use_xy[1]] = cv
    sd_imgs[:, use_xy[0], use_xy[1]] = sd
    if mask_img is False:
        theta_imgs[np.logical_and(np.isnan(theta_imgs), imgs > 0)] = -1
        tau_imgs[np.logical_and(np.isnan(tau_imgs), imgs > 0)] = -1
        cv_imgs[np.logical_and(np.isnan(cv_imgs), imgs > 0)] = -1
        sd_imgs[np.logical_and(np.isnan(sd_imgs), imgs > 0)] = -1
    if mask_img is not False:
        theta_imgs[np.isnan(theta_imgs) * mask_img > 0] = -1
        tau_imgs[np.isnan(tau_imgs) * mask_img > 0] = -1
        cv_imgs[np.isnan(cv_imgs) * mask_img > 0] = -1
        sd_imgs[np.isnan(sd_imgs) * mask_img > 0] = -1
    return theta_imgs, tau_imgs, r2, peak_a[4], peak_a[5], p_time, use_xy, cv_imgs, sd_imgs


def make_peak_img(theta_imgs, mask_imgs=False, idx_t=[24, 24 * 5 + 1]):
    """位相のArrayからピーク画像のArrayを出力.

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
    """Peak画像のArrayにBerつけて画像保存できる形で出力.

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


def img_analysis_pdf(save_folder, tau, r2, cv, sd, distance_center=True, dt=60):
    """作図．諸々の解析の.

    周期の画像を投げられて，Distance_centerからの距離を測る
    """
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    # 解析する対象の時間．
    time = np.arange(0, tau.shape[0], 24 * 60 / dt).astype(np.uint8)
    # 使うデータだけに絞る
    tau = tau[time]
    tau_nan = np.logical_or(np.isnan(tau), tau <= 0)
    cv = cv[time]
    cv_nan = np.logical_or(np.isnan(cv), cv <= 0)
    sd = sd[time]
    sd_nan = np.logical_or(np.isnan(sd), sd <= 0)
    #######################
    # 距離と周期の相関
    #######################
    pp = PdfPages(os.path.join(save_folder, 'tau-distance.pdf'))
    if distance_center is True:
        distance_center = (np.array(tau.shape[1:]).astype(np.float64) * 0.5).astype(np.uint8)
    roops = np.where(np.sum(~tau_nan, axis=(1, 2)) >= 3)[0]
    print(roops)
    for i in roops:
        print(tau[i][~tau_nan[i]].size)
        tau_idx = np.array(np.where(~tau_nan[i]))
        distance = np.linalg.norm((tau_idx.T - distance_center), axis=1)
        title = str(time[i]) + '(h) center[ ' + str(distance_center[0]) + ', ' + str(distance_center[1]) + ']'
        make_hst_fig(save_file=save_folder, x=distance, y=tau[i][~tau_nan[i]], min_x=0, max_x=None, min_y=0, max_y=50, max_hist_x=200, max_hist_y=200, bin_hist_x=200, bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='distance(pixcel)', ylabel='period(h)', pdfpages=pp, box=1, per=True, title=title)
    pp.close()
    #######################
    # 距離と周期の相関
    #######################
    pp = PdfPages(os.path.join(save_folder, 'tau-CV_amp.pdf'))
    cv_tau_nan = np.logical_or(cv_nan, tau_nan)
    roops = np.where(np.sum(~cv_tau_nan, axis=(1, 2)) >= 3)[0]
    for i in roops:
        title = str(time[i]) + '(h)'
        make_hst_fig(save_file=save_folder, x=tau[i][~cv_tau_nan[i]], y=cv[i][~cv_tau_nan[i]], min_x=16, max_x=32, min_y=0, max_y=None, max_hist_x=200, max_hist_y=200, bin_hist_x=200, bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='period(h)', ylabel='cv(amp)', pdfpages=pp, box=1, per=True, title=title)
    pp.close()
    pp = PdfPages(os.path.join(save_folder, 'tau-sd_amp.pdf'))
    sd_tau_nan = np.logical_or(sd_nan, tau_nan)
    roops = np.where(np.sum(~sd_tau_nan, axis=(1, 2)) >= 3)[0]
    for i in roops:
        title = str(time[i]) + '(h)'
        make_hst_fig(save_file=save_folder, x=tau[i][~sd_tau_nan[i]], y=sd[i][~sd_tau_nan[i]], min_x=16, max_x=32, min_y=0, max_y=None, max_hist_x=200, max_hist_y=200, bin_hist_x=200, bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='period(h)', ylabel='cv(amp)', pdfpages=pp, box=1, per=True, title=title)
    pp.close()
    return 0


def img_pixel_theta(folder, mask_folder=False, avg=3, mesh=1, dt=60, offset=0, p_range=12, f_avg=1, f_range=5, save=False, make_color=[22, 28], pdf=False, xlsx=False, distance_center=True, r2_cut=0.5, min_tau=16, max_tau=32):
    """ピクセルごとに二次関数フィッティングにより解析する.

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
        imgs = im.mesh_imgs(folder, mesh)
        if mask_folder is not False:
            mask = im.mesh_imgs(mask_folder, mesh)
        else:
            mask = False
    ##################################
    # 生物発光の画像群から位相を求める
    ##################################
    peak_a = make_theta_imgs(imgs, mask_img=mask, avg=avg, dt=dt, p_range=p_range, f_avg=f_avg, f_range=f_range, offset=offset, r2_cut=r2_cut, min_tau=min_tau, max_tau=max_tau)
    # 位相のデータは醜いので，カラーのデータを返す．
    cv, sd = peak_a[7], peak_a[8]
    color_theta = im.make_colors(peak_a[0], grey=-1, black=np.nan)
    ###################################
    # 保存用に周期を整形
    ###################################
    tau_frond = peak_a[1] == -1
    imgs_tau = (peak_a[1] - make_color[0]) / (make_color[1] - make_color[0]) * 0.7
    nan = ~np.isnan(imgs_tau)
    imgs_tau[nan][imgs_tau[nan] > 0.7] = 0.7
    imgs_tau[nan][imgs_tau[nan] < 0] = 0
    imgs_tau[tau_frond] = -1
    color_legend = np.arange(30) * 0.7 / 30
    color_legend = im.make_color(np.vstack([color_legend, color_legend, color_legend]))
    color_tau = im.make_colors(imgs_tau, grey=-1)
    ####################################
    # 保存用にCVを整形
    ####################
    color_cv = cv / np.nanmax(cv) * 0.7
    color_sd = sd / np.nanmax(sd) * 0.7
    color_cv[cv <= 0], color_sd[sd <= 0] = -1, -1
    color_cv = im.make_colors(color_cv, grey=-1)
    color_sd = im.make_colors(color_sd, grey=-1)
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
        im.save_imgs(os.path.join(save, 'tau' + '_value_' + str(max_tau) + '-' + str(min_tau)), color_tau)
        im.save_imgs(os.path.join(save, 'cv' + '_value_' + str(np.max(cv)) + '-' + str(np.min(cv))), color_cv)
        # im.save_imgs(os.path.join(save, 'sd'), color_sd)
        Image.fromarray(color_legend).save(os.path.join(save, 'color_' + str(make_color[0]) + '-' + str(make_color[0]) + '_cv-' + '{:.2g}'.format(np.nanmax(cv)) + '.png'), compress_level=0)
        Image.fromarray(p_img).save(os.path.join(save, 'peak_img.png'), compress_level=0)
        # phaseを0−1で表したものをcsvファイルに
        np.save(os.path.join(save, 'theta.npy'), peak_a[0])
        np.save(os.path.join(save, 'tau.npy'), imgs_tau)
        if pdf is not False:
            if pdf is True:
                pdf = os.path.join(save, 'pdf')
            else:
                pdf = os.path.join(save, pdf)
            img_analysis_pdf(save_folder=pdf, tau=peak_a[1], r2=peak_a[2], cv=cv, sd=sd, distance_center=distance_center, dt=dt)
        if xlsx is not False:
            writer = pd.ExcelWriter(os.path.join(save, "peak_list.xlsx"))
            peak_a[2].T.to_excel(writer, sheet_name='r2', index=False, header=True)  # 保存
            peak_a[5].T.to_excel(writer, sheet_name='peak_time', index=False, header=True)
            writer.save()
    return 0


def img_fft_nlls(folder, calc_range=[24, 24 * 3], mask_folder=False, avg=1, mesh=1, dt=60, offset=0, save=False, tau_range=[16, 30], pdf=False, xlsx=False):
    """ピクセルごとに二次関数フィッティングにより解析する.

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
            imgs = imgs * mask != 0
    else:
        imgs = im.mesh_imgs(folder, mesh)
        if mask_folder is not False:
            mask = im.mesh_imgs(mask_folder, mesh)
            imgs = imgs * mask != 0
    ##################################
    # 生物発光の画像群から使うデータをcsvからと同じにする
    ##################################
    print(np.sum(np.all(imgs[int(calc_range[0] * 60 / dt):int(calc_range[1] * 60 / dt) + 1] != 0, axis=0)))  # データの存在する場所のインデックスをとってくる)
    use_xy = np.where(np.all(imgs[int(calc_range[0] * 60 / dt):int(calc_range[1] * 60 / dt) + 1] != 0, axis=0))  # データの存在する場所のインデックスをとってくる．
    data = imgs[:, use_xy[0], use_xy[1]]
    print(str(data.shape[1]) + "個のデータを解析します")
    ########################################################
    # 解析
    ########################################################
    data_det, data_det_ampnorm = FFT_nlls.data_norm(data, dt=dt)
    # fft_nlls_det = FFT_nlls.cos_fit(data_det, s=calc_range[0], e=calc_range[1], dt=dt, pdf_plot=False, tau_range=tau_range, pdf=pdf)
    fft_nlls_ampnorm = FFT_nlls.cos_fit(data_det_ampnorm, s=calc_range[0], e=calc_range[1], dt=dt, pdf_plot=False, tau_range=tau_range, pdf=pdf)
    ###################################
    # 周期を画像に戻す．
    ###################################
    tau_dat_img = np.full_like(imgs[0], np.nan, dtype=np.float64)
    tau_ampnorm_img, rae_dat_img, rae_ampnorm_img = np.copy(tau_dat_img), np.copy(tau_dat_img), np.copy(tau_dat_img)
    # tau_dat_img[use_xy], rae_dat_img[use_xy] = fft_nlls_det['tau'], fft_nlls_det['rae']
    tau_ampnorm_img[use_xy], rae_ampnorm_img[use_xy] = fft_nlls_ampnorm['tau'], fft_nlls_ampnorm['rae']

    # tau_dat_img_colure = (tau_dat_img - tau_range[0]) / (tau_range[1] - tau_range[0]) * 0.7
    tau_ampnorm_img_colure = (tau_ampnorm_img - tau_range[0]) / (tau_range[1] - tau_range[0]) * 0.7

    color_legend = np.arange(30) * 0.7 / 30
    color_legend = im.make_color(np.vstack([color_legend, color_legend, color_legend]))
    # tau_dat_img_colure = im.make_color(tau_dat_img_colure, grey=-1)
    tau_ampnorm_img_colure = im.make_color(tau_ampnorm_img_colure, grey=-1)
    ####################################
    # 保存
    ####################
    if save is True:
        save = folder + 'fft_nlls-' + str(calc_range[0]) + '-' + str(calc_range[1]) + '_mesh-' + str(mesh) + '_avg-' + str(avg)
    elif save is not False:
        save = save + 'fft_nlls-' + str(calc_range[0]) + '-' + str(calc_range[1]) + '_mesh-' + str(mesh) + '_avg-' + str(avg)
    if save is not False:
        # np.save(save + 'npy', tau_dat_img)
        np.save(save + '_ampnorm.npy', tau_ampnorm_img)
        np.save(save + '_ampnorm_rae.npy', rae_ampnorm_img)
        # Image.fromarray(tau_dat_img_colure).save(save + '.tif')
        Image.fromarray(tau_ampnorm_img_colure).save(save + '_ampnorm.tif')
    # return tau_dat_img, tau_ampnorm_img, tau_dat_img_colure, tau_ampnorm_img_colure
    return tau_ampnorm_img, tau_ampnorm_img_colure

if __name__ == "__main__":

    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data"))
    # 処理したいデータのフォルダ

    days = (['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX'])
    for day in days:
        frond_folder = os.path.join(day, 'frond')
        folder_list = sorted(glob.glob(os.path.join(frond_folder, '*')))
        for i in folder_list:
            print(i)
            folder = os.path.join(i, 'moved_mask_frond_lum')
            save = os.path.join("result", day, os.path.basename(i))
            img_pixel_theta(folder, avg=3, mesh=1, dt=60, offset=0, p_range=6, f_avg=3, f_range=7, save=save, make_color=[22, 28], xlsx=True, r2_cut=0.5)

    # folder = os.path.join('00data', '170613-LD2LL-ito-MVX', 'frond', 'label-001_n214', 'moved_mask_frond_lum')
    # save = os.path.join('result', '170613-LD2LL-ito-MVX, frond, label-001_n214, fft_nlls')
    # img_fft_nlls(folder, calc_range=[24, 24 * 3], mask_folder=False, avg=1, mesh=1, dt=60, offset=0, save=save, tau_range=[16, 30], pdf=False)

    # img_pixel_theta(folder, avg=3, mesh=3, dt=20, offset=0, p_range=12, f_avg=1, f_range=5, save=save, make_color=[22, 28], xlsx=True)

    # img_pixel_theta(folder, avg=3, mesh=3, dt=20, offset=0, p_range=12, f_avg=1, f_range=15, save=save, make_color=[22, 28], xlsx=True)
