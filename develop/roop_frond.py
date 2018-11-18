# -*- coding: utf-8 -*-
"""Roop about frond."""

# import numpy as np
import os
import glob
# import sys
# from phase_from_img import img_to_mesh_phase
from _181029_diff import phase_diff_all


def roop_frond(func, folder_path, res='*', **kwargs):
    """Roop frond."""
    # を回す．folder内のresを含むファイル名をfuncの第一引数に."""
    # funcの引数は dt=60 のように普段通り指定．(**kwargs内に入る).
    for dir_path in sorted(glob.glob(os.path.join(folder_path, res))):
        print(dir_path)
        func(dir_path, **kwargs)


def roop_day(func, days, folder_path, res='*', **kwargs):
    """Roop day."""
    for day in days:
        print(day)
        roop_frond(func, os.path.join(day, folder_path), res=res, **kwargs)


if __name__ == '__main__':
    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data"))
    # 処理したいデータのフォルダ
    days = (['170215-LL2LL-MVX', '170613-LD2LL-ito-MVX', '170829-LL2LL-ito-MVX'])

    # roop_frond(img_to_mesh_phase, folder_path=day, avg=3, mesh=1, dt=60, peak_avg=3, p_range=12, fit_range=5, save_folder=False, pdf_save=False)
    for temp in days:
        day = os.path.join(temp, 'frond_180730')
        roop_frond(phase_diff_all, folder_path=day, phase_path='small_phase_mesh1_avg3.npy', save_path='phase_diff.pdf', point=list([36, 4 * 24, 7 * 24]), scale=0.033)
