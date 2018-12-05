# -*- coding: utf-8 -*-
"""phase diff - distance."""

import os
# import glob
# import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# 下２行は，リモートで操作している場合
import matplotlib as mpl
# mpl.use('Agg')


def phase_diff(phase_img, pp=False, scale=1, title='', x_max=3.5, x_range=0.05):
    """Phase diff - distance from image."""
    x, y = np.where(~np.isnan(phase_img))  # 位相が定義されている座標
    xy = np.array((x, y)).T
    phase_d = phase_img[x, y]  # 上に対応する位相のリスト
    size = np.size(phase_d)  # 使うデータの個数
    idx = range(np.size(phase_d))
    tri_dx = np.triu_indices(size, 1)  # 上三角行列のIndex
    xx, yy = np.meshgrid(idx, idx)

    dis = np.linalg.norm(xy[xx[tri_dx]] - xy[yy[tri_dx]], axis=1)  # 距離
    dis = dis * scale
    diff = np.abs(phase_d[xx[tri_dx]] - phase_d[yy[tri_dx]])  # 位相差
    diff[diff > 0.5] = 1 - diff[diff > 0.5]

    if pp is not False:  # pdfの保存
        fig = plt.figure(figsize=(11.69, 8.27), dpi=100)
        sc_bottom = sc_left = 0.1
        sc_width = 0.85
        sc_height = 0.65
        space = 0.01
        hst_height = 0.15
        ####################################################
        # ボックスプロットをする
        ax = plt.axes([sc_bottom, sc_left, sc_width, sc_height])
        x_avg = np.arange(0, x_max, x_range)  # 村中博論に合わせて0.05mm感覚で
        diff_box = []
        for c, i in enumerate(x_avg):
            idx = (dis < (i + x_range)) * (dis >= i)
            diff_box.append(diff[idx])  # ボックスプロットは要素をタプルで欲しがる
        whiskerprops = {'linestyle': 'solid', 'linewidth': -1, 'color': 'k'}
        ax.boxplot(diff_box, positions=x_avg + x_range * 0.5, showmeans=True, meanline=True, whis='range', widths=x_range, whiskerprops=whiskerprops, showcaps=False)
        ax.set_ylim(0, 0.5)

        ax.set_xlim(0, x_max)
        ax.set_xlabel(r'cell-to-cell distance ($\mu m $)')
        ax.set_ylabel(r'phase diffrence ($rad / \pi $)')
        ax.set_xticks(np.arange(0, x_max, 0.5))  # x軸の調整．トリッキーなので…
        ax.set_xticklabels(np.arange(0, x_max, 0.5))  # 同上

        ########################################################
        # ヒストグラムを上につける
        ax_x = plt.axes([sc_left, sc_bottom + sc_height + space, sc_width, hst_height])
        ax_x.set_xlim(0, x_max)
        ax_x.set_xticklabels([])
        ax_x.hist(dis, bins=int(x_max / x_range), histtype='stepfilled')
        ax_x.set_ylim(bottom=0)

        ax_x.set_title(title)
        plt.savefig(pp, format='pdf')
        fig.clf()
    return diff, dis


def phase_diff_all(dir_path='', phase_path='small_phase_mesh1_avg3.npy', save_path='phase_diff.pdf', point=list([36, 4 * 24, 7 * 24]), scale=1, dbg=False):
    """Loop phase_diff_all."""
    # scale 33um らしい．僕の修論によると．
    folder = os.path.dirname(os.path.join(dir_path, save_path))
    if os.path.exists(folder) is False and folder != '':
        os.makedirs(folder)
    phase = np.load(os.path.join(dir_path, phase_path))
    phase = phase[point]
    if dbg is True:
        phase[~np.isnan(phase)] = np.random.rand(np.sum(~np.isnan(phase)))
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = 'sans-serif'
    pp = PdfPages(os.path.join(dir_path, save_path))
    for i in range(len(point)):
        if np.sum(~np.isnan(phase[i])) > 2:
            phase_diff(phase[i], pp=pp, scale=scale, title=str(point[i] / 24) + 'day')
    plt.close()
    pp.close()
    return 0


if __name__ == '__main__':
    # os.getcwd() # これでカレントディレクトリを調べられる
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya',
                          'Labo', 'keisan', 'python', '00data'))
    phase_diff_all(dir_path='', phase_path='170215-LL2LL-MVX/frond_180730/label-001_268-432_n184/small_phase_mesh1_avg3.npy', save_path='randam_phase_diff.pdf', point=list([36, 4 * 24, 7 * 24]), scale=0.033, dbg=True)
