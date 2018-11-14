# -*- coding: utf-8 -*-
import numpy as np
import math
import os
import cv2
import glob
import sys
import matplotlib.pyplot as plt
import image_analysis as im
import label2folder
import transform
import make_phase_img
import phase2R

if __name__ == '__main__':
    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    # dataフォルダ
    # day = ('./170215-LL2LL-MVX')
    # day = ('./170613-LD2LL-ito-MVX')
    day = ('./170829-LL2LL-ito-MVX')

    # 解析データのフォルダ
    label_folder = day + '/edit_raw/label_img/'
    lum_folder = day + '/edit_raw/lum_min_img/'
    out_folder = day + '/frond'
    # 出力先フォルダ
    label2folder.label2frond(label_folder, out_folder, lum_folder)

    frond_folder = day + '/frond'
    for i in glob.glob(frond_folder + '/*'):
        print(i)
        calc_img = im.read_imgs(i + '/mask_frond')
        mask_lum_img = im.read_imgs(i + '/mask_frond_lum')
        lum_img = im.read_imgs(i + '/frond_lum')
        calc_img, mask_lum_img, lum_img, warps = transform.imgs_transform(calc_img, mask_lum_img, lum_img)
        np.save(i + '/warps.npy', warps)
        warps.resize((2*(calc_img.shape[0]-1), 3))
        np.savetxt(i + '/warps.csv', warps, delimiter=',')
        im.save_imgs(i + '/moved_mask_frond', calc_img)
        im.save_imgs(i + '/moved_mask_frond_lum', mask_lum_img)
        im.save_imgs(i + '/moved_frond_lum', lum_img)

    frond_folder = day + '/frond'
    for i in glob.glob(frond_folder + '/*'):
        print(i)
        # 解析データのフォルダ
        data_folder = i + '/moved_mask_frond_lum'
        save_folder = i
        color, imgs_phase = make_phase_img.img_to_mesh_phase(data_folder, avg=1, mesh=1, dT=60, peak_avg=3, p_range=12, fit_range=5, save_folder=save_folder)

    data_file = '/phase_mesh1_avg1.npy'
    for i in glob.glob(frond_folder + '/*'):

        frond = im.read_imgs(os.path.join(i, 'mask_frond_lum'))
        area = np.empty(frond.shape[0])
        for j in range(frond.shape[0]):
            area[j] = np.count_nonzero(frond[j])
        print(area)
        # 解析データのフォルダ
        data = np.load(i + data_file)
        save_folder = day + '/result/R/' + (data_file.lstrip('/')).rstrip('.npy') + str('_') + i.split('/')[-1]
        if os.path.exists(day+'/result/R') is False:
            os.makedirs(day + '/result/R')
        phase2R.phase2R_plt(data, area, save_folder)
