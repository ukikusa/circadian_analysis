import numpy as np
import os
import itertools
import glob
import sys
import pandas as pd
import image_analysis as im
import cv2
from make_phase_img import img_to_mesh_phase
from frond_traker import frond_traker
from label2folder import label2frond


if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    folder = "160721_LLtoLL_CCA1_MVX/data_light"
    dT = 30
    if 0:
        light_img = im.read_imgs(folder)
        light_img8 = im.bit1628(light_img)
        img = cv2.imread(folder + '_thresh_edit.tif', cv2.IMREAD_GRAYSCALE)
        print(img.shape)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        cv2.imwrite(folder + '_thresh_end.tif', img)

        img_lum = im.read_imgs('160721_LLtoLL_CCA1_MVX/data', color=False)
        img_frond = cv2.imread(folder + '_thresh_end.tif', cv2.IMREAD_GRAYSCALE)
        frond_lum = np.zeros_like(img_lum)
        frond_lum[:, img_frond == 0] = img_lum[:, img_frond == 0]
        print(np.average(frond_lum[frond_lum != 0]))
        frond_lum[frond_lum > 10000] = np.average(frond_lum[frond_lum != 0])
        im.save_imgs('160721_LLtoLL_CCA1_MVX/frond_lum', frond_lum)

    if 0:
        dT = 30
        color, imgs_phase = img_to_mesh_phase('160721_LLtoLL_CCA1_MVX/frond_lum', avg=3, mesh=1, dT=dT, peak_avg=3, p_range=24, fit_range=7, save_folder='color')
    ##################################################################
    ################### ここからUBQの前処理 ##########################
    ##################################################################
    folder = "160728_UBQ/data_light"
    dT = 30
    if 0:
        # 背景除去
        light_img = im.read_imgs(folder)
        edit_img = cv2.imread(folder + '_edit.tif', cv2.IMREAD_GRAYSCALE + cv2.IMREAD_ANYDEPTH)
        img_rm_bk = edit_img - light_img
        img_rm_bk[edit_img < light_img] = 0
        im.save_imgs(folder + '_rm_bk', img_rm_bk)
        # 二値化
        im_rm_bk = im.read_imgs(folder + '_rm_bk')
        im_rm_bk = im.bit1628(im_rm_bk)
        im.save_imgs(folder + '_rm_bk_8bit', im_rm_bk)
        im_thresh = np.zeros_like(im_rm_bk)
        im_thresh[im_rm_bk >= 30] = 255
        im.save_imgs(folder + '_thresh', im_thresh)
        # 二値化を手作業で修正
        im_thresh = im.read_imgs(folder + '_thresh')
        for i in np.arange(im_thresh.shape[0]):
            im_thresh = cv2.morphologyEx(im_thresh, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))
            im_thresh = cv2.morphologyEx(im_thresh, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        im_rm_bk = im.read_imgs(folder + '_rm_bk')
        im_rm_bk[im_thresh == 0] = 0
        im.save_imgs(folder + '_thresh_rm_bk', im_rm_bk)

    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'python', '00data', "160728_UBQ"))
    if 0:
        # ラベル付け
        im_edit = im.read_imgs('data_light_edit_01')

        im_edit[im_edit != 0] = 255
        imgs_sort, imgs_centroids, imgs_stats_sort, label_out = frond_traker(im_edit, min_size=500)
        im.save_imgs('label_imgs_tmp', imgs_sort)
        # ラベル付の修正
        im_edit = im.read_imgs('label_imgs_tmp2')
        im_edit[im_edit != 0] = 255
        imgs_sort, imgs_centroids, imgs_stats_sort, label_out = frond_traker(im_edit, min_size=500)
        im.save_imgs('label_imgs_tmp3', imgs_sort)
        label_out_save = pd.DataFrame(label_out, columns=np.arange(1, label_out.shape[1] + 1))
        label_out_save = label_out_save.replace(0, np.nan)
        label_out_save.to_csv('label_tmp.csv')

    # ラベル2が欲しいフロンドなので，取り出す

    # 解析データのフォルダ
    label_folder = 'label_img'
    img = im.read_imgs(label_folder)
    edit_img = np.zeros_like(img)
    edit_img[img == 2] = 255
    for i in np.arange(edit_img.shape[0]):
        edit_img = cv2.morphologyEx(edit_img, cv2.MORPH_CLOSE, np.ones((7, 7), dtype=np.uint8))
        edit_img = cv2.morphologyEx(edit_img, cv2.MORPH_OPEN, np.ones((7, 7), dtype=np.uint8))
        edit_img = cv2.morphologyEx(edit_img, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
        edit_img = cv2.morphologyEx(edit_img, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8))

    im.save_imgs(label_folder + 'frond_2', edit_img)
    label_folder = 'label_imgfrond_2'
    lum_folder = 'lum_min_img'
    out_folder = 'frond'
    # 出力先フォルダ
    label2frond(label_folder, out_folder, lum_folder, size=300)
    # imgs = im.read_imgs(data_folder)
