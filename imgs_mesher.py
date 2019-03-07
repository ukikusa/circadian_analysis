# -*-coding: utf-8 -*-
"""フロンド内，ピクセルごとの解析を行う.
現行の出力は，24時間幅のcv,sd(ともに振幅), 周期，位相の画像．"""

import analyser.image_analysis as im

import os
import sys
# ディレクトリを import の探索パスに追加
sys.path.append('/hdd1/Users/kenya/Labo/keisan/python_sorce/circadian_analysis/analyser')


os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
# カレントディレクトリの変更．
#########################
# パラメータ
#########################
folder = os.path.join('00data', 'nakamura_mask')  # 入力(画像の入ったフォルダ)を指定する．
save = os.path.join('result', 'nakamura', 'mesh')  # 出力先を指定する．
# save = True  # 自動命名で保存．
dt = 60  # 何分間隔で撮影したかを指定．

img = im.mesh_imgs(folder, mesh=3)
im.save_imgs(save, img)