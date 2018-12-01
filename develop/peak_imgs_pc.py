# -*-coding: utf-8 -*-
"""Receive the time series image of the phase and reacquire the time when the peak was reached."""

import os

import image_analysis as im
import numpy as np
from PIL import Image  # Pillowの方を入れる．PILとは共存しない


def peak_img(dir_path='', phase_path='phase.npy', mask_path='mask_frond', save_path='peak_time_img', peak_n=0, peak_per=0, idx_t=[24, 24 * 4 + 1]):
    """Receive the time series image of the phase and reacquire the time when the peak was reached."""
    # phaseは．.npyファイル．mask_pathはフロンド以外の画素が0の画像．
    # 本当は，ピークを求めるタイミングで保存しておけばいい話．
    phase = np.load(os.path.join(dir_path, phase_path))  # .npyファイルをロード
    phase = phase[idx_t[0]: idx_t[1]]  # 必要な部分だけ切り出し
    diff = np.diff(phase, n=1, axis=0)  # どの画像とどの画像の間でピークが来ているかを計算
    peak_time_img = np.zeros_like(diff, dtype=np.uint8)
    if mask_path is not False:
        mask_img = im.read_imgs(os.path.join(dir_path, mask_path))  # フロンドのある場所を描写
        mask_img = mask_img[idx_t[0]: idx_t[1]]
        peak_time_img[mask_img[:-1] != 0] = 50
    else:
        peak_time_img[~np.isnan(phase[:-1])] = 50
    # peakの来ている場所を描写
    peak_time_img[diff < 0] = 255
    peak_sum = np.sum(peak_time_img == 255, axis=(1, 2))
    frond_sum = np.sum(peak_time_img != 0, axis=(1, 2))
    idx = np.where(peak_sum >= peak_n)[0]
    idx = np.where(peak_sum >= frond_sum * peak_per)[0]
    if save_path is not False:
        im.save_imgs(os.path.join(dir_path, save_path), peak_time_img, idx=idx)
    return peak_time_img, peak_sum, frond_sum


def peak_pc_img(parent_path, child_path, save_path=False, phase_path='phase.npy', mask_path='mask_frond', per_ber=10, idx_t=[24, 24 * 3 + 1]):
    """Peak time comparison between parent and child."""
    key = {'phase_path': phase_path, 'mask_path': mask_path, 'save_path': False, 'peak_n': 0, 'peak_per': 0, 'idx_t': idx_t}  # peak_img に投げるキーワード．
    m = 10  # 画像間の白線． 引数にしてもいい．

    def make_ber(img, per_ber, frond, xsize, m=5):
        """Output a ber of the percentage of pixels that are peaks."""
        if per_ber == 0 or False:
            return img
        ber_len = xsize - m * 2  # berの長さ
        ber = np.tile(np.arange(ber_len), (n, per_ber, 1))  # ber の下地
        idx = (ber.T <= peak.astype(np.float64) * ber_len / frond).T  # 塗りつぶす場所
        ber[idx] = 255
        ber[~idx] = 50
        img[:, -m - per_ber:-m, -m + per_ber:-m] = ber
        return img
    #################
    # 親フロンドの描写．
    parent, peak, frond = peak_img(dir_path=parent_path, **key)  # 取り込み
    n, xsize, ysize = parent.shape[0:3]  # 画像サイズ
    parent = make_ber(parent, per_ber, frond, xsize)  # ピークと判定されたピクセル数のバーを出す．
    parent = np.reshape(parent.transpose(1, 2, 0), (ysize, -1), 'F')  # 画像を横に並べる．
    for path in child_path:  # 子フロンドを追加する．
        child, peak, frond = peak_img(dir_path=path, **key)
        child = make_ber(child, per_ber, frond, xsize)  # ピークと判定されたピクセル数のバーを出す．
        child = np.reshape(child.transpose(1, 2, 0), (ysize, -1), 'F')   # 画像を横に並べる
        parent = np.vstack((parent, child))  # 画像を縦に並べる
    parent[:m] = 255
    print(np.hsplit(parent, int(n / 24))[1].shape)
    parent = np.vstack(np.hsplit(parent, int(n / 24)))
    if save_path is not False:
        Image.fromarray(parent).save(os.path.join(save_path))
    return parent


if __name__ == '__main__':
    # os.getcwd() # これでカレントディレクトリを調べられる
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
    # カレントディレクトリの変更．
    ################
    # pathの指定
    ####################
    day = os.path.join('00data', '170613-LD2LL-ito-MVX', 'frond_180730')

    phase_path = os.path.join('small_phase_mesh1_avg3.npy')
    mask_path = os.path.join('moved_mask_frond')
    save_path = os.path.join('_181114', 'result', 'frond_001.tif')
    parent_path = os.path.join(day, 'label-001_239-188_n214')
    child_path = [os.path.join(day, 'label-002_261-255_n214'), os.path.join(day, 'label-003_279-159_n214')]
    ###############
    # 実行
    ###############
    peak_pc_img(parent_path, child_path, save_path, phase_path, mask_path, idx_t=[24, 24 * 3 + 1])
