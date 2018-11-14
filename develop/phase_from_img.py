# -*- coding: utf-8 -*-
"""Phase from image and fig."""

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
import pandas as pd
# from _171020_make_period_img import make_period_imgs
import scipy.stats
import textwrap
import sys
import matplotlib as mpl
mpl.use('Agg')


def make_hst_fig(save_file, x, y, min_x=0, max_x=None, min_y=0, max_y=None, max_hist_x=None, max_hist_y=None, bin_hist_x=100, bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], pdfpages=False, avg=False, xlabel='', ylabel='', box=False, per=True):
    """Create a graph with histogram.  You can also find correlation coefficients or plot box plots."""
    # Min_x-max_y:散布図の範囲．max_hist:度数分布の最大値，bin:度数分布の分割.
    x, y = x.astype(np.float64), y.astype(np.float64)
    folder = os.path.dirname(save_file)
    if os.path.exists(folder) is False:
        os.makedirs(folder)
    ###############################
    # graph position
    sc_bottom = sc_left = 0.1
    sc_width = sc_height = 0.65
    space = 0.01
    hst_height = 0.2
    ####################################################
    fig = plt.figure(1, figsize=(6, 4), dpi=100)
    ax = plt.axes([sc_bottom, sc_left, sc_width, sc_height])
    xy = np.vstack([x, y])
    z = scipy.stats.gaussian_kde(xy)(xy)  # 色を決める．
    ax.scatter(x, y, c=z, marker='.', s=1)
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    #######################################
    # pearsonr
    if per is True:
        r, p = scipy.stats.pearsonr(x, y)
        r_p_text = textwrap.wrap('Pearson’s correlation coefficient: ' + '{:.2g}'.format(r), 30)
        r_p_text.extend(textwrap.wrap('2-taild p-value: ' + '{:.2g}'.format(p), 30))
        fig.text(sc_left + sc_width + space, sc_bottom + sc_height + space, '\n'.join(r_p_text), fontsize=6)
    ##########################
    # 平均とSEをプロット
    if avg is not False:
        min_x, max_x = ax.get_xlim()
        # x_avg = np.arange(min_x, max_x, avg)
        x_avg = np.arange(max(np.floor(np.min(x)), min_x), min(np.ceil(np.max(x)), max_x), avg, dtype=np.float64)  # 平均を取る
        y_avg, y_se = [], []
        for i in x_avg:
            idx = (x < (i + avg)) * (x >= i)
            y_avg.append(np.average(y[idx]))
            # y_se.append(np.std(y[idx],ddof=1)/np.sqrt(y[idx].size)) #se
            y_se.append(np.std(y[idx], ddof=1))  # SD
        ax.errorbar(x_avg + avg * 0.5, y_avg, yerr=y_se, fmt='r.', ecolor='k', elinewidth=0.5)
    # box prot
    if box is not False:
        ax_b = ax.twiny()
        min_x, max_x = ax.get_xlim()
        x_avg = np.arange(min_x, max_x, box, dtype=np.float64)  # 平均を取る
        y_avg = []
        for i in x_avg:
            idx = (x < (i + box)) * (x >= i)
            y_avg.append(y[idx])
        whiskerprops = {'linestyle': 'solid', 'linewidth': -1, 'color': 'k'}
        ax_b.boxplot(y_avg, positions=(x_avg + box * 0.5), showmeans=True, meanline=True, whis='range', widths=box, whiskerprops=whiskerprops, showcaps=False)
        ax_b.set_xticks([])  # x軸の調整
    # ここまで
    ###################
    if xticks is not False:
        ax.set_xticks(xticks)
    if yticks is not False:
        ax.set_yticks(yticks)
    if xticklabels != []:
        ax.set_xticklabels(xticklabels)
    if yticklabels != []:
        ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ###################
    # hisutoglam
    ###################
    ax_x = plt.axes([sc_left, sc_bottom + sc_height + space, sc_width, hst_height])
    ax_x.set_ylim(0, max_hist_x)
    ax_x.set_xlim(ax.get_xlim())
    ax_x.set_xticklabels([])
    ax_x.hist(x, bins=bin_hist_x, histtype='stepfilled', range=ax.get_xlim())
    ax_y = plt.axes([sc_left + sc_width + space, sc_bottom, hst_height, sc_width])
    ax_y.set_ylim(ax.get_ylim())
    ax_y.set_xlim(0, max_hist_y)
    ax_y.set_yticklabels([])
    ax_y.hist(y, bins=bin_hist_y, histtype='stepfilled', range=ax.get_ylim(), orientation='horizontal')
    if pdfpages is False:
        fig.savefig(save_file)
        plt.close()
    else:
        plt.savefig(pdfpages, format='pdf')
        plt.clf()


def make_phase_imgs(imgs, avg, dT=60, peak_avg=3, p_range=6, fit_range=4, csv_save=False, offset=0):
    data = imgs.reshape([imgs.shape[0], imgs.shape[1] * imgs.shape[2]])
    phase_imgs = np.empty(imgs.shape, dtype=np.float64)
    phase_imgs[:] = np.nan
    period_imgs = np.copy(phase_imgs)
    time = np.arange(imgs.shape[0], dtype=np.float64) * dT / 60 + offset
    use_xy = np.where(np.sum(imgs, axis=0) != 0)
    label = use_xy[0].astype(str).astype(object) + ',' + use_xy[1].astype(str).astype(object)
    use_index = np.where(np.sum(data, axis=0) != 0)[0]
    # ここでピーク抽出やグラフ作成を終わらす
    peak_a = pa.phase_analysis(data[:, use_index], avg=avg, dT=dT, p_range=p_range, fit_range=fit_range, time=time)
    peak_time = pd.DataFrame(peak_a[0], columns=label)
    r2 = pd.DataFrame(peak_a[3], columns=label)
    phase_imgs[:, use_xy[0], use_xy[1]] = peak_a[2]
    period_imgs[:, use_xy[0], use_xy[1]] = peak_a[6]
    if csv_save is not False:
        peak_time.to_csv(csv_save)
    return phase_imgs, period_imgs, r2, peak_a[4], peak_a[5], use_xy


def mesh_img(folder, mesh=5):
    """Make the image mash by taking the average in the vicinity."""
    # 画像のフォルダから画像を全部読み込んできて，全てメッシュ化してしまおう！
    img = im.read_imgs(folder)
    # 以下メッシュ化するよ！
    meshed = np.empty((img.shape[0], int(img.shape[1] / mesh), int(img.shape[2] / mesh)))
    for i in range(int(img.shape[1] / mesh)):
        for j in range(int(img.shape[2] / mesh)):
            meshed[::, i, j] = img[::, i * mesh:(i + 1) * mesh, j * mesh:(j + 1) * mesh].mean(axis=(1, 2))
    return meshed


def img_to_mesh_phase(folder, avg, mesh=1, dT=60, peak_avg=3, p_range=12, fit_range=5, save_folder=False, pdf_save=False, make_color=[22, 28], save_color=False):
    # 上の全部まとめたった！！
    # メッシュ化
    if mesh == 1:
        data = im.read_imgs(folder)
    else:
        data = mesh_img(folder, mesh)
    # 生物発光の画像群から位相を求める
    peak_a = make_phase_imgs(imgs=data, avg=avg, dT=dT, peak_avg=peak_avg, p_range=p_range, fit_range=fit_range)
    imgs_phase = peak_a[0]
    # 位相のデータは醜いので，カラーのデータを返す．
    color_phase = im.make_colors(imgs_phase)
    imgs_period = (peak_a[1]-make_color[0])/(make_color[1]-make_color[0])*0.8
    color_period = im.make_colors(imgs_period)
    if save_folder is not False:
        # color 画像の保存
        im.save_imgs(os.path.join(save_folder, 'small_phase_color' + '_mesh' + str(mesh) + '_avg' + str(avg)), color_phase)
        im.save_imgs(os.path.join(save_folder, 'small_period_color' + '_mesh' + str(mesh) + '_avg' + str(avg)), color_period)
        # phaseを0−1で表したものをcsvファイルに
        np.save(save_folder + '/small_phase' + '_mesh' + str(mesh) + '_avg' + str(avg) + '.npy', imgs_phase)
        np.save(os.path.join(save_folder, 'small_period' + '_mesh' + str(mesh) + '_avg' + str(avg) + '.npy'), imgs_period)
    if pdf_save is not False:
        x = np.linalg.norm(np.asarray(peak_a[5])-80, axis=0).repeat(np.asarray(peak_a[2]).shape[0])
        y = np.asarray(peak_a[2]).reshape(len(x), order='F')
        x, y = x[~np.isnan(y)], y[~np.isnan(y)]
        max_x, max_y = 45, 1
        make_hst_fig(save_file=pdf_save, x=x, y=y, max_x=max_x, max_y=max_y, max_hist=500, bin_hist=100)
    return color_phase, imgs_phase, peak_a[2]


# dataはdata,avgで移動平均．前後p_range分より高い点を暫定ピークに．その時の移動平均がpeak_avg．fit_rangeでフィッティング

if __name__ == "__main__":
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))  # カレントディレクトリの移動．ググって．ないしは，下のフォルダ指定で絶対パス指定をする．

    # 解析データのフォルダ
    data_folder = os.path.join('.', 'small_moved_mask_frond_lum')  # 助長かも．データのあるフォルダを指定して．
    save_folder = '.'
    out = img_to_mesh_phase(data_folder, avg=3, mesh=1, dT=60, peak_avg=3, p_range=12, fit_range=5, save_folder=save_folder, pdf_save=os.path.join(save_folder, 'tmp.pdf'))

    sys.exit('正常に終了')
    ################
    # roop回したければ以下の通り
    ####################
    days = ['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX']
    for day in days:
        frond_folder = day + '/frond'
        for i in sorted(glob.glob(frond_folder + '/*')):
            print(i)
            # 解析データのフォルダ
            data_folder = i + '/small_moved_mask_frond_lum/'
            save_folder = i
            out = img_to_mesh_phase(data_folder, avg=3, mesh=1, dT=60, peak_avg=3, p_range=12, fit_range=5, save_folder=save_folder, pdf_save=os.path.join(save_folder, 'tmp.pdf'))
