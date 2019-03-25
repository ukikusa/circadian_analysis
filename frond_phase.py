# -*-coding: utf-8 -*-
"""フロンド内，ピクセルごとの解析を行う.
現行の出力は，24時間幅のcv,sd(ともに振幅), 周期，位相の画像．"""

import os
import sys
# ディレクトリを import の探索パスに追加
sys.path.append('/hdd1/Users/kenya/Labo/keisan/python_sorce/circadian_analysis/analyser')
from phase_from_img import img_pixel_theta


os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
# カレントディレクトリの変更．
#########################
# パラメータ
#########################
folder = os.path.join('00data', '170215-LL2LL-MVX', 'frond_180730', 'label-001_268-432_n184', 'small_moved_mask_frond_lum')
# 入力(画像の入ったフォルダ)を指定する．
save = os.path.join('result', '170215-LL2LL', "label-001")  # 出力先を指定する．
# save = True  # 自動命名で保存．
dt = 60  # 何分間隔で撮影したかを指定．

img_pixel_theta(folder, avg=5, dt=dt, mesh=1, offset=0, p_range=7, f_avg=5, f_range=6, save=save, make_color=[20, 30], xlsx=True, pdf=True, distance_center=True, r2_cut=0.0)
