# -*- coding: utf-8 -*-


import glob
import itertools
import os
import sys

import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd


def img_transformECC(tmp_img, new_img, motionType=1):
    '''Estimate rotation matrix by transformECC.
    
    Args:
        tmp_img: Template image
        new_img: Moving image
        motionType: [0: tanslation, 1:Euclidean, 2:affine, 3:homography] (default: {1})
        centroid: [description] (default: {[0,0]})
    
    Returns:
        [description]
        [type]
    '''
    if np.max(tmp_img) > 255 or np.max(new_img) > 255:
        tmp_img = im.bit1628(tmp_img)
        new_img = im.bit1628(new_img)
    tmp_img, new_img = tmp_img.astype(np.uint8), new_img.astype(np.uint8)
    # 上記に合わせた，WarpMatrixを指定　warpは結果を格納するところ
    warp = np.eye(2, 3, dtype=np.float32)
    size = new_img.shape    # 出力画像のサイズを指定
    # 移動を計算
    try:  # エラーでたら…
        cv2.findTransformECC(new_img, tmp_img, warp, motionType=motionType)
        temp = 0
    except:
        print('移動に失敗した画像があります')
        temp = 1
        warp = np.eye(2, 3, dtype=np.float32)
    # 元画像に対して，決めた変換メソッドで変換を実行
    out = cv2.warpAffine(new_img, warp, size, flags=cv2.INTER_NEAREST)
    out[np.nonzero(out)] = np.max(out) - temp
    return out, warp, temp


def imgs_transformECC(calc_img, centroids=False, motionType=1):
    # 移動補正をまとめた．
    warps = np.zeros((calc_img.shape[0] - 1, 2, 3), dtype=np.float32)
    warps[:, 0, 0], warps[:, 1, 1] = 1, 1
    if centroids is False:
        centroids = np.zeros((calc_img.shape[0]))
    else:
        centroids = np.modf(centroids)[0]  # 少数部分のみ
    tmp = np.max(calc_img, axis=(1, 2))
    roop = np.where((tmp[:-1] * tmp[1:]) != 0)[0]
    for i in roop:  # 全部黒ならする必要ない．
        calc_img[i + 1][np.nonzero(calc_img[i + 1])] = np.max(calc_img[i])
        calc_img[i + 1], warps[i], _ = img_transformECC(calc_img[i], calc_img[i + 1], motionType=motionType, centroid=centroids[i])
    return calc_img, warps, roop


def imgs_transformECC_ver2(calc_imgs, motionType=1):
    '''Motion correction based on a specific image.
    
    Args:
        calc_img: [description]
        centroids: [description] (default: {False})
        motionType: [description] (default: {1})
    
    Returns:
        [description]
        [type]
    '''
    # 移動補正をまとめた．
    warps = np.zeros((calc_imgs.shape[0] - 1, 2, 3), dtype=np.float32)
    warps[:, 0, 0], warps[:, 1, 1] = 1, 1
    tmp = np.max(calc_imgs, axis=(1, 2))
    roop = np.where((tmp[:-1]*tmp[1:]) != 0)[0]
    try_r = 0
    tmp_img = calc_imgs[0]
    for i in roop:  # 全部黒ならする必要ない．
        if try_r == 1:  # 直前の画像が移動補正に失敗した場合
            calc_imgs[i + 1][np.nonzero(calc_imgs[i + 1])] = np.max(calc_imgs[i])
            tmp_img = calc_imgs[i]
        calc_imgs[i + 1], warps[i], try_r = img_transformECC(tmp_img, calc_imgs[i + 1], motionType=motionType)
    return calc_img, warps, roop


if __name__ == '__main__':
    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data"))
    # 処理したいデータのフォルダ
    days = (['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX'])
    # days = (['./170829-LL2LL-ito-MVX'])
    for day in days:
        frond_folder = day + '/frond'
        folder_list = sorted(glob.glob(frond_folder + '/*'))
        trance = pd.DataFrame(index=[], columns=['first'])
        for i in folder_list:
            print(i)
            calc_img = im.read_imgs(i + '/mask_frond')
            centroids = np.loadtxt(os.path.join(i, "centroids.csv"), delimiter=",")
            mask_lum_img = im.read_imgs(i + '/mask_frond_lum')
            lum_img = im.read_imgs(i + '/frond_lum')
            trance.at[i, 'first'] = np.sum(calc_img[0] != 0)
            trance.at[i, 'last'] = np.sum(calc_img[-1] != 0)

            print('ECC')
            move_img, warps, roops = imgs_transformECC(calc_img)
            trance.at[i, 'ECC_moved'] = np.sum((move_img[0]!=0) != (move_img[-1]!=0))/(np.sum(move_img[-1] != 0))*0.5

            ############### ECCver2 ################
            print('ECC')
            move_img, warps, roops = imgs_transformECC_ver2(calc_img)
            trance.at[i, 'ECC_moved_2'] = np.sum((move_img[0]!=0) != (move_img[-1]!=0))/(np.sum(move_img[-1] != 0))*0.5
            ############## ECCと重心 #################
            # print('ECCと重')
            # move_img, warps, roops = imgs_transformECC(calc_img, centroids=centroids)
            # trance.at[i, 'cg_move'] = np.sum(move_img[0] != move_img[-1])/(np.sum(move_img[-1] != 0))*0.5
            imgs = img_transform(warps, roops, mask_lum_img, lum_img)
            # np.save(i + '/warps.npy', warps)
            # warps.resize(((calc_img.shape[0] - 1), 3))
            # np.savetxt(i + '/warps.csv', warps, delimiter=',')
            # im.save_imgs(i + '/moved_mask_frond', move_img)
            # im.save_imgs(i + '/moved_mask_frond_lum', imgs[0])
            # im.save_imgs(i + '/moved_frond_lum', imgs[1])
        trance.to_csv(os.path.join(day, 'tranc.csv'))
