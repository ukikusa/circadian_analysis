# -*- coding: utf-8 -*-

import os

# import sys

import cv2
import numpy as np
import pandas as pd

# import image_analysis as im
import analyser.image_analysis as im


def label_img2frond(img, lumi_img, centroids, size=160):
    centroids_int = centroids.astype(np.int16)  # データ喪失が嫌なので重心を整数に．
    # ここから，画像を切り抜く場所を決めるけど，頭悪い．なんとかしたい．
    img_rightup = (centroids_int - size / 2).astype(np.int16) + 1
    img_leftdown = img_rightup + size
    [R, U, L, D] = np.zeros((4, img.shape[0]), dtype=np.uint8)
    L[:], D[:] = size, size
    R[img_rightup[:, 1] < 0] = -img_rightup[img_rightup[:, 1] < 0, 1]
    L[img_leftdown[:, 1] >= 512] = -img_rightup[img_leftdown[:, 1] >= 512, 1] - 512
    U[img_rightup[:, 0] < 0] = -img_rightup[img_rightup[:, 0] < 0, 0]
    D[img_leftdown[:, 0] >= 512] = -img_rightup[img_leftdown[:, 0] >= 512, 0] - 512
    img_rightup[img_rightup < 0] = 0
    img_leftdown[img_leftdown >= 512] = 512
    # ここから，画像を切り抜いて代入．
    new_img = np.zeros((img.shape[0], size, size), dtype=np.uint16)
    new_lumi_img = np.zeros((img.shape[0], size, size), dtype=np.uint16)
    # ここのループもなくしたい．
    for i in range(img.shape[0]):
        new_img[i, R[i] : L[i], U[i] : D[i]] = img[
            i,
            img_rightup[i, 1] : img_leftdown[i, 1],
            img_rightup[i, 0] : img_leftdown[i, 0],
        ]
        if lumi_img is not False:
            new_lumi_img[i, R[i] : L[i], U[i] : D[i]] = lumi_img[
                i,
                img_rightup[i, 1] : img_leftdown[i, 1],
                img_rightup[i, 0] : img_leftdown[i, 0],
            ]
    if lumi_img is False:
        new_lumi_img = False
    return new_img, new_lumi_img


def label2frond(folder, out_folder, lum_folder=False, size=160):
    # label付されたフォルダを取り込み，出力は
    data = im.read_imgs(folder)
    s_idx, e_idx, frond_idx = [], [], []
    # data = im.bit1628(data)
    if lum_folder is not False:
        lum_data = im.read_imgs(lum_folder)
    else:
        lum_data = False
    # label毎にループを回す．
    for i in np.arange(1, np.max(data) + 1):
        # 解析可能フロンドが以下の枚数以上ならば解析します．
        if np.sum(np.any(data == i, axis=(1, 2))) >= 48:
            # 箱作り
            stats = np.zeros((data.shape[0], 5), dtype=int)
            centroids = np.zeros((data.shape[0], 2), dtype=np.float64)
            frond_img = np.zeros_like(data)

            # i番目のコロニーだけを抽出．
            frond_img[data == i] = 255
            if lum_folder is not False:
                lum_frond = np.zeros_like(lum_data)
                lum_frond[data == i] = lum_data[data == i]
            else:
                lum_frond = False
            frond_index = np.where(np.sum(frond_img, axis=(1, 2)))
            s_idx.append(frond_index[0][0])
            e_idx.append(frond_index[0][-1])
            for j in frond_index[0]:
                # 画像毎に重心等の情報を取得する．
                # 一枚目 statsは端の座標が4個，最後が面積． centroidsは重心
                tmp = cv2.connectedComponentsWithStats(frond_img[j], connectivity=4)
                _, stats[j], centroids[j] = tmp[0], tmp[2][1], tmp[3][1]

            # フロンドを切り出す
            img_mask, lum_img_mask = label_img2frond(
                frond_img, lum_frond, centroids, size=size
            )
            img, lum_img = label_img2frond(data, lum_data, centroids, size=size)

            # 保存
            frond_idx.append("label-" + str(i).zfill(3))
            save_folder = os.path.join(out_folder, frond_idx[-1])
            im.save_imgs(save_folder, img, file_name="frond.tif")
            im.save_imgs(save_folder, img_mask, file_name="mask_frond.tif")
            centroids = pd.DataFrame(centroids, columns=["centroids_x", "centroids_y"])
            stats = pd.DataFrame(
                stats, columns=["up_x", "up_y", "under_x", "under_y", "area"]
            )
            pd.concat([centroids, stats], axis=1).to_csv(
                os.path.join(save_folder, "stats.csv")
            )
            if lum_folder is not False:
                im.save_imgs(save_folder + "/mask_frond_lum", lum_img_mask)
                im.save_imgs(save_folder + "/frond_lum", lum_img)
    sr = pd.DataFrame({"s_idx": s_idx, "e_idx": e_idx}, index=frond_idx)
    sr.to_csv(os.path.join(os.path.split(out_folder)[0], "frond_number.csv"))


if __name__ == "__main__":
    # os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    os.chdir(
        os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data")
    )
    # dataフォルダ
    days = ["170613-LD2LL-ito-MVX", "./170215-LL2LL-MVX", "./170829-LL2LL-ito-MVX"]
    for day in days:
        # 解析データのフォルダ
        label_folder = day + "/edit_raw/label_img/"
        lum_folder = day + "/edit_raw/lum_min_img/"
        out_folder = day + "/frond"
        # 出力先フォルダ
        label2frond(label_folder, out_folder, lum_folder)
        # imgs = im.read_imgs(data_folder)
        # color = standardization(imgs)
