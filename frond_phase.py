# -*-coding: utf-8 -*-
"""Today."""

import os
import sys
# ディレクトリを import の探索パスに追加
sys.path.append('/hdd1/Users/kenya/Labo/keisan/python_sorce/develop')
from phase_from_img import img_pixel_theta


os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
# カレントディレクトリの変更．
#########################
# パラメータ
#########################
folder = os.path.join('00data', '170613-LD2LL-ito-MVX', 'frond_180730', 'label-001_239-188_n214', 'small_moved_mask_frond_lum')
save = os.path.join('_181204', 'test')  # 保存先のファイル
save = True  # 自動命名で保存．

img_pixel_theta(folder, avg=5, mesh=1, dt=60, offset=0, p_range=7, f_avg=5, f_range=6, save=save, make_color=[22, 28], xlsx=True, pdf=True, distance_center=True, r2_cut=0.5)