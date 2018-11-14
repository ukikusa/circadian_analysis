# -*- coding: utf-8 -*-
"""fft to fig."""

import os
import glob
# import sys
import numpy as np
import pandas as pd
from phase_from_img import make_hst_fig
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# 下２行は，リモートで操作している場合
# import matplotlib as mpl
# mpl.use('Agg')

# 仮バージョン


def distance_pdf(dir_path='', name_path='46_118_name.csv', period_path='22_96_Ampnorm_dataFFTnlls_data_0_75.73FFTnlls_results.csv', save_path='tmp.pdf', rae=1, delimiter=',', period_min=22, period_max=28):
    """Make pdf."""
    folder = os.path.dirname(save_path)
    if os.path.exists(folder) is False:
        os.makedirs(folder)
    # タブ区切りはdelimitterを','以外に
    # RAEが指定以下のデータのみプロット
    if delimiter == ',':
        name = pd.read_csv(os.path.join(dir_path, name_path))
        period = pd.read_csv(os.path.join(dir_path, period_path))
    else:
        name = pd.read_table(os.path.join(dir_path, name_path))
        period = pd.read_table(os.path.join(dir_path, period_path))
    # print(period.values[:, 'Tau'])
    name_v = name.values[1, :]
    period_v = period.ix[:, 'Tau'].values
    rae_v = period.ix[:, 'RAE'].values
    amp_v = period.ix[:, 'Amp']
    rae_nan = ~np.isnan(rae_v)
    period_v, name_v, rae_v, amp_v = period_v[rae_nan], name_v[rae_nan], rae_v[rae_nan], amp_v[rae_nan]
    rae_idx = rae_v <= rae
    period_v_rae, name_v_rae, amp_v_rae = period_v[rae_idx], name_v[rae_idx], amp_v[rae_idx]
    if period_v_rae.size <= 5:
        return 0
    plt.rcParams['font.size'] = 9
    plt.rcParams['font.family'] = 'sans-serif'
    pp = PdfPages(os.path.join(save_path))
    # 周期と距離のプロット
    make_hst_fig(save_file=save_path, x=name_v_rae, y=period_v_rae, min_x=0, max_x=None, min_y=20, max_y=30, max_hist_x=200, max_hist_y=200, bin_hist_x=int(
        np.max(name_v)), bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='distance(pixcel)', ylabel='period(h)', pdfpages=pp, avg=1)
    # raeと周期
    make_hst_fig(save_file=save_path, x=period_v, y=rae_v, min_x=20, max_x=30, min_y=0, max_y=1, max_hist_x=100, max_hist_y=500,
                 bin_hist_x=200, bin_hist_y=100, xticks=False, yticks=False, xlabel='period(h)', ylabel='rae', pdfpages=pp)
    # raeと距離のプロット
    make_hst_fig(save_file=save_path, x=name_v, y=rae_v, min_x=0, max_x=None, min_y=0, max_y=1, max_hist_x=200, max_hist_y=200, bin_hist_x=int(np.max(name_v)),
                 bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='distance(pixcel)', ylabel='rae', pdfpages=pp, box=2)
    # period
    make_hst_fig(save_file=save_path, x=name_v_rae, y=amp_v_rae, min_x=0, max_x=None, min_y=0, max_y=None, max_hist_x=200, max_hist_y=200, bin_hist_x=int(np.max(name_v)),
                 bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='distance(pixcel)', ylabel='fft_amp', pdfpages=pp, box=2)
    make_hst_fig(save_file=save_path, x=period_v_rae, y=amp_v_rae, min_x=22, max_x=30, min_y=0, max_y=None, max_hist_x=200, max_hist_y=200, bin_hist_x=int(np.max(name_v)), bin_hist_y=100, xticks=False, yticks=False, xticklabels=[], yticklabels=[], xlabel='period (h)', ylabel='fft_amp', pdfpages=pp, box=0.2)
    plt.clf()
    pp.close()
    return 0


if __name__ == '__main__':
    # os.getcwd() # これでカレントディレクトリを調べられる
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya',
                          'Labo', 'keisan', 'python', 'R_fft'))
    # name_path = '22_94_name.csv'
    # period_path = '22_94_Ampnorm_dataFFTnlls_data_0_75.73/FFTnlls_results.csv'
    # rae = 0.5  # raeがこれ以下のもののみ保存
    # save_path = os.path.join('_181010_2', 'result', 'distance' + '_rae-' + str(rae) + '.pdf')
    ###############
    # 実行 下の関数のコメントアウトを外す
    #######
    # distance_pdf(dir_path=dir_name, name_path=name_path, period_path=period_path, save_path=save_path, rae=rae, delimiter=',', period_min=22, period_max=28)
    # カレントディレクトリの変更．
    ################
    # pathの指定
    ####################
    days = ['170215-LL2LL-MVX', '170613-LD2LL-ito-MVX', '170829-LL2LL-ito-MVX']
    rae = 0.3  # raeがこれ以下のもののみ保存
    for day in days:
        fronds = sorted(glob.glob(os.path.join(day, '*')))
        for frond in fronds:
            print(frond)
            name_list = sorted(glob.glob(os.path.join(frond, 'fft', '*raw_name.csv')) +
                               (glob.glob(os.path.join(frond, 'fft', '*norm_name.csv'))))
            # name_list = sorted(glob.glob(os.path.join(frond, 'fft', '*2_name.csv'))+(glob.glob(os.path.join(frond, 'fft', '*4_name.csv')))+(glob.glob(os.path.join(frond, 'fft', '*8_name.csv'))))
            period_list = sorted(glob.glob(os.path.join(
                frond, 'fft', '*_dataFFTnlls_data_*')))
            for i in range(len(period_list)):
                name_path = name_list[i]
                period_path = os.path.join(
                    period_list[i], 'FFTnlls_results.txt')
                save_path = os.path.join(frond, os.path.basename(name_path)[
                                         :-9] + '-rae-' + str(rae) + '.pdf')
                print(save_path)
                ###############
                # 実行 下の関数のコメントアウトを外す
                #######
                distance_pdf(dir_path='', name_path=name_path, period_path=period_path,
                             save_path=save_path, rae=rae, delimiter='\t', period_min=22, period_max=28)
    # ピークに当たる部分がpeak_n以上の画像だけ保存．n=0で全部保存
    # インデックスエラーとか言われたら，コンマ区切り，タブ区切りがちゃんとそうなっているか調べる．csvファイルだからといってコンマ区切りになっているとは限らない．
