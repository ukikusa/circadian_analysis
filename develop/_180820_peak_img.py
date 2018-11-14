# -*-coding: utf-8 -*-
"""Receive the time series image of the phase and reacquire the time when the peak was reached."""

import os
# import glob
# import sys
import numpy as np
# import pandas as pd
import image_analysis as im
from roop_frond import roop_day


def peak_img(dir_path='', phase_path='phase.npy', mask_path='mask_frond', save_path='peak_time_img', peak_n=0, peak_per=0):
    """Receive the time series image of the phase and reacquire the time when the peak was reached."""
    # phaseは．.npyファイル．mask_pathはフロンド以外の画素が0の画像．
    # 本当は，ピークを求めるタイミングで保存しておけばいい話．
    phase = np.load(os.path.join(dir_path, phase_path))  # .npyファイルをロード
    # どの画像とどの画像の間でピークが来ているかを計算
    diff = np.diff(phase, n=1, axis=0)
    peak_time_img = np.zeros_like(diff, dtype=np.uint8)
    # フロンドのある場所を描写
    mask_img = im.read_imgs(os.path.join(dir_path, mask_path))
    peak_time_img[mask_img[:-1] != 0] = 125
    # peakの来ている場所を描写
    peak_time_img[diff < 0] = 255
    peak_sum = np.sum(peak_time_img == 255, axis=(1, 2))
    frond_sum = np.sum(peak_time_img != 0, axis=(1, 2))
    idx = np.where(peak_sum >= peak_n)[0]
    idx = np.where(peak_sum >= frond_sum * peak_per)[0]
    if save_path is not False:
        im.save_imgs(os.path.join(dir_path, save_path), peak_time_img, idx=idx)
    return peak_time_img, peak_sum, frond_sum


def peak_pc_img(parent_path, child_path, save_path=False, phase_path='phase.npy', mask_path='mask_frond', per_ber=3):
    """Peak time comparison between parent and child."""
    key = {'phase_path': phase_path, 'mask_path': mask_path, 'save_path': False, 'peak_n': 0, 'peak_per': 0}  # peak_img に投げるキーワード．
    m = 2  # 画像につける枠の幅．

    def make_ber(img, per_ber, frond, xsize, m=m + 1):
        """Output a ber of the percentage of pixels that are peaks."""
        if per_ber == 0 or False:
            return img
        ber_len = xsize - m * 2  # berの長さ
        ber = np.tile(np.arange(ber_len), (n, per_ber, 1))  # ber の下地
        idx = (ber.T <= peak.astype(np.float64) * ber_len / frond).T  # 塗りつぶす場所
        ber[idx] = 255
        ber[~idx] = 125
        img[:, m:-m, -m + per_ber:-m] = ber
        return img
    #################
    # 親フロンドの描写．
    parent, peak, frond = peak_img(dir_path=parent_path, **key)  # 取り込み
    n, xsize, ysize = parent.shape[0:3]  # 画像サイズ
    parent = make_ber(parent, per_ber, frond, xsize)  # ピークと判定されたピクセル数のバーを出す．
    parent = np.reshape(parent.transpose(1, 2, 0), (ysize, -1), 'F')  # 画像を横に並べる．
    parent[0:m, :], parent[-1 - m:-1, :], parent[:, 0:m], parent[:, -1 - m:-1] = 255, 255, 255, 255  # 枠

    for path in child_path:  # 子フロンドを追加する．
        child = peak_img(dir_path=path, **key)
        child = make_ber(parent, per_ber, frond, xsize)  # ピークと判定されたピクセル数のバーを出す．
        child = np.reshape(child.transpose(1, 2, 0), (ysize, -1), 'F')   # 画像を横に並べる
        child[0:m, :], child[-1 - m:-1, :], child[:, 0:m], child[:, -1 - m:-1] = 255, 255, 255, 255  # 枠
        parent = np.vstack((parent, child))  # 画像を縦に並べる
    if save_path is not False:
        im.save_imgs(save_path, parent)
    return parent


if __name__ == '__main__':
    # os.getcwd() # これでカレントディレクトリを調べられる
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    # カレントディレクトリの変更．
    ################
    # pathの指定
    ####################
    phase_path = os.path.join('small_phase_mesh1_avg3.npy')
    mask_path = os.path.join('moved_mask_frond')
    save_path = os.path.join('peak_tmp_img')
    peak_n = 20  # peakが来たピクセル数がこれ以上の画像のみ保存
    ###############
    # 実行 下の関数のコメントアウトを外す
    #######
    # peak_img(phase_path, mask_path, save_path, peak_n = 20)
    # ピークに当たる部分がpeak_n以上の画像だけ保存．n=0で全部保存

    # 以下，コメントアウトして(上野用)
    days = ['170215-LL2LL-MVX', '170613-LD2LL-ito-MVX', '170829-LL2LL-ito-MVX']
    frond_path = os.path.join('frond_180730')
    roop_day(peak_img, days=days, folder_path=frond_path, phase_path=phase_path, mask_path=mask_path, save_path=save_path, peak_n=peak_n)
    # roop_frond(peak_img, folder_path=frond_path, phase_path=phase_path, mask_path=mask_path, save_path=save_path, peak_n=peak_n)
