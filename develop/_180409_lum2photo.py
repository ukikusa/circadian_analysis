import numpy as np
# import matplotlib.pyplot as plt
import os
import sys
import glob
import datetime
import image_analysis as im
import pandas as pd
from _180112_csv2fig import csv2fig_all
from _180113_lum2photo import img2photon
import image_analysis as im
from label2folder import label2frond
from _171213_flond_data_pandas import frond_folder2data
from transform import imgs_transform
from _171005 import img_for_bio
from make_phase_img import img_to_mesh_phase
from phase2R import phase2Rs
from phase2R import frond_plt
import cv2


if __name__ == '__main__':
    # os.chdir(os.path.join("/Users", "kenya", "keisan", "python", "00data"))
    os.chdir(os.path.join("/hdd1", "kenya", "Labo", "keisan", "python", "00data"))
    days = "./160728_UBQ"
    dt = 30
    offset = 0
    dark = 5196
    frond_folder = os.path.join(days, "frond")
    img2photon(frond_folder, dt=dt, offset=offset, dark=dark, save=True)

    ############################################
    ################ 総発光量 ##################
    ############################################
    data_file = "2018-04-09-frond_photo_sum.csv"
    avg = 3
    ymax = 2000000
    loc = "out right"
    loc = "def"

    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc)

    ############################################
    ################# 面積 #####################
    ############################################
    data_file = "2018-04-09-frond_area.csv"
    avg = 3
    ymax = 60000
    loc = "out right"

    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)

    ############################################
    ################# 平均 #####################
    ############################################
    data_file = "2018-04-09-frond_photo_avg.csv"
    avg = 3
    ymax = 40
    loc = "out right"

    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)

    ################# peak_move #############
    ###############################
    day = './170728-UBQ'

    dt = 30

    # label_folder = os.path.join(day, "edit_raw", "label_img")
    # lum_folder = os.path.join(day, "edit_raw", "lum_min_img")
    # frond_folder = os.path.join(day, "frond")
    # 出力先フォルダ

    # ラベルごとのフォルダを作成
    # if os.path.exists(frond_folder) is False:
    #     label2frond(label_folder, frond_folder, lum_folder)

    # flondの発光量，面積，平均を求める．
    #      frond_folder2data(frond_folder, time=dt, offset=offset, save=True)

    # 回転
    for i in sorted(glob.glob(frond_folder + '/*')):
        print(i)
        # 普通の方
        calc_img = im.read_imgs(os.path.join(i, 'mask_frond'))
        mask_lum_img = im.read_imgs(os.path.join(i, 'mask_frond_lum'))
        lum_img = im.read_imgs(os.path.join(i, 'frond_lum'))
        calc_img2, mask_lum_img2, lum_img2 = np.empty_like(calc_img), np.empty_like(mask_lum_img), np.empty_like(lum_img)
        calc_img2, mask_lum_img2, lum_img2, warps = imgs_transform(calc_img, mask_lum_img, lum_img)
        np.save(i + '/warps.npy', warps)
        warps.resize((2 * (calc_img.shape[0] - 1), 3))
        np.savetxt(i + '/warps.csv', warps, delimiter=',')
        im.save_imgs(i + '/moved_mask_frond', calc_img2)
        im.save_imgs(i + '/moved_mask_frond_lum', mask_lum_img2)
        im.save_imgs(i + '/moved_frond_lum', lum_img2)
        del warps

        # エッジ使う方
        # カーネル
        kernel = np.ones((3, 3), np.uint8)
        for j in range(calc_img.shape[0]):
            calc_img[j] = cv2.morphologyEx(calc_img[j], cv2.MORPH_GRADIENT, kernel)  # エッジ
        calc_img2, mask_lum_img2, lum_img2, warps = imgs_transform(calc_img, mask_lum_img, lum_img)
        np.save(i + '/warps.npy', warps)
        warps.resize((2 * (calc_img.shape[0] - 1), 3))
        np.savetxt(i + '/warps.csv', warps, delimiter=',')
        im.save_imgs(i + '/moved_edge_mask_frond', calc_img2)
        im.save_imgs(i + '/moved_edge_mask_frond_lum', mask_lum_img2)
        im.save_imgs(i + '/moved_edge_frond_lum', lum_img2)

    # 最も長い部分だけを残す．
    for i in sorted(glob.glob(frond_folder + '/*')):
        print(i)
        img = im.read_imgs(i + '/moved_mask_frond')
        lum_img = im.read_imgs(i + '/moved_mask_frond_lum')
        small_img = img_for_bio(img)
        im.save_imgs(i + '/small_moved_mask_frond', small_img)
        lum_img[small_img == 0] = 0
        im.save_imgs(i + '/small_moved_mask_frond_lum', lum_img)
        # エッジの方のやっとこ
        img = im.read_imgs(i + '/moved_edge_mask_frond')
        lum_img = im.read_imgs(i + '/moved_edge_mask_frond_lum')
        small_img = img_for_bio(img)
        im.save_imgs(i + '/small_edge_moved_mask_frond', small_img)
        lum_img[small_img == 0] = 0
        im.save_imgs(i + '/small_edge_moved_mask_frond_lum', lum_img)

    # 解析開始．カラーの画像を作る．ピークを見つけてピークの図を作る
    for i in sorted(glob.glob(frond_folder + '/*')):
        print(i)
        # 解析データのフォルダ
        data_folder = i + '/small_moved_mask_frond_lum/'
        save_folder = i
        color, imgs_phase = img_to_mesh_phase(data_folder, avg=3, mesh=1, dt=dt, peak_avg=3, p_range=12, fit_range=5, save_folder=save_folder, pdf_save=save_folder)

    # 同期率求める
    data_file = 'small_phase_mesh1_avg3.npy'
    for i in sorted(glob.glob(frond_folder + '/*')):
        print(i)
        frond = im.read_imgs(os.path.join(i, 'small_moved_mask_frond_lum', ''))
        area = np.empty(frond.shape[0])
        lum_sum = np.empty(frond.shape[0])
        for j in range(frond.shape[0]):
            area[j] = np.count_nonzero(frond[j])
            lum_sum[j] = np.sum(frond[j])
            # 解析データのフォルダ
        data = np.load(os.path.join(i, data_file))
        save_folder = os.path.join(day, 'result', 'small_R_avg3', (data_file.lstrip('/')).rstrip('.npy') + str('_') + i.split('/')[-1])
        if os.path.exists(day + '/result/small_R_avg3') is False:
            os.makedirs(day + '/result/small_R_avg3')
        r, number, euler = phase2Rs(data)
        np.save(i + '/small_R.npy', r)
        np.save(i + '/small_number.npy', number)
        frond_plt(r, number, area, lum_sum, save_folder)
