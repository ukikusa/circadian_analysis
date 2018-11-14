# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys
import image_analysis as im
import peak_analysis as pa

import shutil


def npy2csv(a):
    a=0
    return 0


if __name__=='__main__':
    # os.chdir('/media/kenya/HD-PLFU3/kenya/171013_jikan/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    # 処理したいデータのフォルダ
    # day = ('./170829-LL2LL-ito-MVX')
    days = ['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX']
    # 解析データのフォルダ
    for day in days:
        frond_folder = os.path.join(day, 'frond')
        for i in sorted(glob.glob(os.path.join(frond_folder, '*'))):
            print(i)
            # 画像の取り込み．移動へ意見
            text = "small_phase_mesh1_avg3"
            folder_img_small = 'small_moved_mask_frond_lum'
            img = im.read_imgs(os.path.join(i, folder_img_small))
            # for j in np.arange(img_nonavg.shape[1]):
            #     for k in np.arange(img_nonavg.shape[2]):
            #        img[:, j, k] = pa.moving_avg(img_nonavg[:, j, k], 1)
            # とりあえず座標の設定と画像の整形
            x = np.array(list(np.arange(img.shape[2]))*img.shape[1])
            y = np.repeat(np.arange(img.shape[1]), img.shape[2])
            img = img.reshape((img.shape[0], -1))
            # 使うピクセルを抽出
            frond_pixel = np.any(img != 0, axis=0)
            frond_x, frond_y, frond_img = x[frond_pixel], y[frond_pixel], img[:, frond_pixel]
            # 距離の計算
            pixel_linalg = np.linalg.norm((np.array([frond_x, frond_y])-80), axis=0)
            out = np.empty((img.shape[0]+3, frond_x.shape[0]))
            out[0], out[1], out[2], out[3:] = frond_x, frond_y, pixel_linalg, frond_img
            len_sort = np.argsort(pixel_linalg)
            out = out[:, len_sort]
            # os.mkdir(os.path.join(i, 'result_171016'))
            for_fft = np.arange(0, 24*3 + int(out.shape[0]/24-5)*24, 24)
            shutil.rmtree(os.path.join(i, 'result_171016'))
            os.mkdir(os.path.join(i, 'result_171016'))
            for j in for_fft:
                out_tmp = out[j+2:j+24*3+2]
                if np.sum(np.all(out_tmp != 0, axis=0)) != 0:
                    out_little = np.empty((out_tmp.shape[0]+2, np.sum(np.all(out_tmp != 0, axis=0))))
                    out_little[2:] = out_tmp[:, np.all(out_tmp != 0, axis=0)]
                    out_little[:3] = out[:3, np.all(out_tmp != 0, axis=0)]
                    if out_little[:, out_little[2]<=21] != [] and out_little[:, out_little[2]>21] != []:
                        np.savetxt(os.path.join(i, 'result_171016', str(j) + 'in.csv'), out_little[:, out_little[2]<=21], delimiter=',')
                        np.savetxt(os.path.join(i, 'result_171016', str(j) + 'out.csv'), out_little[:, out_little[2]>21], delimiter=',')