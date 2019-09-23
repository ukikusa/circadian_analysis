# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import glob
import sys
import math
import datetime

import pandas as pd


sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import image_analysis as im

# frondを追跡する．入力は背景とエッジが0になっている画像．
# 出力はラベル付されたsingle chanel画像．特定のフロンドには同じ値が与えられる．


def frond_shaping(imgs):
    # 8bit の二値化画像に．
    imgs[imgs != 0] = 255
    imgs = imgs.astype(np.uint8)
    # ここから一枚ごとの処理をするよ！
    for i in range(imgs.shape[0]):
        # まず穴を消す
        img = imgs[i, :, :]
        label, img_label, img_stats, other = cv2.connectiedcomponentsWithStats(
            img, connectivity=4
        )
    return 0


def trash_frond(label, img, stats, centroids, min_size=20):  # 小さい領域を削除する
    img_out, stats, centroids = np.zeros_like(img), np.array(stats), np.array(centroids)
    leave = np.where(stats[:, -1] > min_size)[0]
    stats, centroids = stats[leave], centroids[leave]
    for i, j in enumerate(leave):
        img_out[img == j] = i
    label = leave.shape[0]
    return label, img_out, stats, centroids


def frond_traker(imgs, min_size=20):
    imgs[imgs != 0] = 255
    imgs = imgs.astype(np.uint8)
    # アウトプット用の箱作り
    imgs_sort = np.zeros_like(imgs)
    # 一枚目 statsは端の座標が4個，最後が面積． centroidsは重心
    # 0にはバックグラウンドが入っているので要注意．
    label, imgs_sort[0], stats_tmp, centroids_tmp = cv2.connectedComponentsWithStats(
        imgs[0], connectivity=4
    )
    label, imgs_sort[0], stats_tmp, centroids_tmp = trash_frond(
        label, imgs_sort[0], stats_tmp, centroids_tmp, min_size=min_size
    )
    # 箱を作る．それぞれ，ラベルに対応する位置に代入する．
    # label_out は手作業によるラベル付け直しのため．
    label_out = np.tile(np.arange(1, 255, dtype=np.uint8), (imgs.shape[0], 1))
    imgs_stats_sort = np.empty((imgs.shape[0], 255, 5), dtype=int)
    imgs_centroids = np.zeros((imgs.shape[0], 255, 2), dtype=np.float64)
    imgs_stats_sort[0, 0 : label - 1], imgs_centroids[0, 0 : label - 1] = (
        stats_tmp[1:],
        centroids_tmp[1:],
    )
    label_out[0, :] = label_out[0, :] * np.in1d(label_out[0, :], imgs_sort[0,])
    for i in range(1, imgs.shape[0]):
        print(i)
        # 前のフロンドと次のフロンドを格納
        img_front, img_after = imgs_sort[i - 1], np.zeros_like(imgs_sort[i - 1])
        # 前のフロンドにあるラベル
        label_number = np.sort(np.unique(img_front))[1:]
        # 追跡する画像をラベル付
        label, img, stats, centroids = cv2.connectedComponentsWithStats(
            imgs[i], connectivity=4
        )
        label, img, stats, centroids = trash_frond(
            label, img, stats, centroids, min_size=min_size
        )
        # 重心間の距離を図る．
        # 0の位置は背景の重心が入っているので無視（無視してないけど大丈夫？．
        centroids = np.array(centroids[1:])  # おまじない
        stats = np.array(stats[1:])
        # print(centroids)
        min_index = np.empty(label_number.shape[0], dtype=int)
        min_distance = np.empty(label_number.shape[0], dtype=np.float64)
        for j, k in enumerate(label_number):
            #  # kのフロンドに対する次の画像のフロンドの距離
            distance = np.linalg.norm(imgs_centroids[i - 1, k - 1] - centroids, axis=1)
            # distance = np.linalg.norm(imgs_stats_sort[i-1,k-1,0:2]-stats[:,0:2], axis=1)
            # j番目の位置にkのフロンドに対して距離が一番短いフロンドのインデックスを格納，そして距離を収納．
            min_index[j], min_distance[j] = np.argmin(distance), np.min(distance)
        # フロンドのラベルを前の画像と合わせる．
        for j, k in enumerate(min_index):
            # 同じインデクスを複数回参照していたら距離が短い方を取る．
            same_frond = np.where(min_index == k)[0]
            if min_distance[j] == min(min_distance[same_frond]):
                # 距離が短い方だけ以下の処理をする
                # 画像にフロンド番号を代入し，重心ボックスに重心を入れる．
                img_after[img == k + 1] = label_number[j]
                imgs_centroids[i, label_number[j] - 1] = centroids[k]
                imgs_stats_sort[i, label_number[j] - 1] = stats[k]
        # 前の画像になかったフロンド
        new_label = np.sort(np.unique(img[img_after == 0]))[1:]
        for j, k in enumerate(new_label):
            print(j, k)
            img_after[img == k] = j + label_number[-1] + 1
            imgs_centroids[i, j + label_number[-1]] = centroids[k - 1]
            imgs_stats_sort[i, j + label_number[-1]] = stats[k - 1]
        # 画像の格納庫に格納
        imgs_sort[i] = img_after
        label_out[i, :] = label_out[i, :] * np.in1d(label_out[i, :], img_after)
    # label_out[label_out == 0] = np.nan
    return imgs_sort, imgs_centroids, imgs_stats_sort, label_out


if __name__ == "__main__":
    os.chdir(
        os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data")
    )
    # day = "./170613-LD2LL-ito-MVX"
    day = "./170829-LL2LL-ito-MVX"
    # day = "./170215-LL2LL-MVX"
    # ラベリングが終わった画像を取り込む．エッジと背景が0・それ以外は0で無い値なら良い
    # 白黒でもカラーでもOK
    frond_folder = os.path.join(day, "edit_raw", "div_ito_edit.tif")
    # frond_folder = day + '/edit_raw/label_img_ito'
    imgs = im.read_imgs(frond_folder)
    imgs_sort, imgs_centroids, imgs_stats_sort, label_out = frond_traker(
        imgs, min_size=5
    )
    im.save_imgs(os.path.join(day, "edit_raw"), imgs_sort, "label_imgs_tmp.tif")
    label_out_save = pd.DataFrame(
        label_out, columns=np.arange(1, label_out.shape[1] + 1)
    )
    label_out_save = label_out_save.replace(0, np.nan)
    label_out_save.to_csv(os.path.join(day, "edit_raw", "label_tmp.csv"))
