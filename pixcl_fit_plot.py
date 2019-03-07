# -*-coding: utf-8 -*-
"""フロンド内，ピクセルごとの解析を行う.
現行の出力は，24時間幅のcv,sd(ともに振幅), 周期，位相の画像．"""

import analyser.image_analysis as im
import analyser.peak_analysis as pa
import analyser.make_figure as make_figure
import numpy as np
import os
import sys
# ディレクトリを import の探索パスに追加
sys.path.append('/hdd1/Users/kenya/Labo/keisan/python_sorce/circadian_analysis/analyser')


os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
# カレントディレクトリの変更．
#########################
# パラメータ
#########################
folder = os.path.join('00data', '170613-LD2LL-ito-MVX', 'frond_180730', 'label-001_239-188_n214', 'small_moved_mask_frond_lum')  # 入力(画像の入ったフォルダ)を指定する．
save = os.path.join('result', 'plot_pdf_test', 'tmp.pdf')  # 出力先を指定する．
# save = True  # 自動命名で保存．
dt = 60  # 何分間隔で撮影したかを指定．

index = [[80, 80], [60, 60]]

f_range = 5

img = im.read_imgs(folder)
time = np.arange(img.shape[0]) * dt / 60
data = np.vstack([img[:, idx[0], idx[1]] for idx in index]).T
label = np.hstack(['[' + str(idx[0]) + ',' + str(idx[1]) + ']' for idx in index]).T


peak_t, peak_v, d_theta, r2, peak_point, func, d_tau = pa.phase_analysis(data, avg=3, dt=60, p_range=12, f_avg=1, f_range=f_range, offset=0, time=False, r2_cut=False, min_tau=16, max_tau=32)
make_figure.multi_plot(time, data, save, peak=peak_point, func=func, r=f_range, label=label, y_min=0, y_max=None, plt_x=2, plt_y=1, size_x=11.69, size_y=8.27)
