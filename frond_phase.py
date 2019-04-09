# -*-coding: utf-8 -*-
"""フロンド内，ピクセルごとの解析を行う.現行の出力は，24時間幅のcv,sd(ともに振幅), 周期，位相の画像．"""

import os
import sys
# ディレクトリを import の探索パスに追加
sys.path.append(os.getcwd())
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analyser.phase_from_img import img_fft_nlls
from analyser.phase_from_img import img_pixel_theta


os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
# カレントディレクトリの変更．
#########################
# パラメータ
#########################
# folder = os.path.join('00data', 'nakamura', '190302_CCA1_uraまとめ', 'Result of CCA1_ura_2000_masked')
# 入力(画像の入ったフォルダ)を指定する．
# save = os.path.join('result', 'nakamura', "190302_CCA1_ura_mesh")  # 出力先を指定する．
save = True  # 自動命名で保存．
# dt = 20  # 何分間隔で撮影したかを指定．
# folder = gui_path.dir_select(initialdir=__file__)
img_pixel_theta(folder, avg=5, dt=dt, mesh=1, offset=0, p_range=7, f_avg=5, f_range=6, save=save, make_color=[20, 30], xlsx=True, pdf=True, distance_center=True, r2_cut=0.0)
img_fft_nlls(folder, calc_range=[24, 24 * 3], mask_folder=False, avg=1, mesh=1, dt=60, offset=0, save=False, tau_range=[16, 30], pdf=False)
# img_pixel_theta(folder, avg=5, dt=dt, mesh=3, offset=0, p_range=7, f_avg=1, f_range=9, save=save, make_color=[20, 30], xlsx=True, pdf=True, distance_center=True, r2_cut=0.0)
