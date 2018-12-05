# -*- coding: utf-8 -*-


import glob
import itertools
import os
import sys

import cv2

import image_analysis as im

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



def get_warp(trans, rotation=False, size=120):
    size = np.array(size)
    warp = np.eye(2, 3, dtype=np.float32)
    center = size * 0.5 + trans
    if rotation is not False:
        warp = rotation
    warp[:, 2] = [(1 - warp[0, 0]) * center[0] - warp[0, 1] * center[1], warp[0, 1] * center[0] + (1 - warp[0, 0]) * center[1]]
    return warp


def add_warp(trans, warp, size=120):
    size = np.array(size)
    center = size * 0.5 + trans
    if warp[0, 0] != 1:
        center = np.array([0, 0])+np.array(trans)
        warp[:, 2] = warp[:, 2] + [(1 - warp[0, 0]) * center[0] - warp[0, 1] * center[1], warp[0, 1] * center[0] + (1 - warp[0, 0]) * center[1]]
    return warp


def img_transformECC(tmp_img, new_img, motionType=1, centroid=[0,0]):
    # 移動補正のためのの設定
    # warp_type　変換メソッド指定
    # 0, cv2.MOTION_TRANSLATION 並進運動
    # 1, cv2.MOTION_EUCLIDEAN ユーグリッド変換．　回転，移動
    # 2, cv2.MOTION_AFFINE     AFFINE変換．　回転，移動，拡大
    # 3, cv2.MOTION_HOMOGRAPHY 射影変換．　形も変わる場合．
    # 上記に合わせた，変換関数とWarpMatrixを指定　warpは結果を格納するところ
    if np.max(tmp_img) > 255 or np.max(new_img) > 255:
        tmp_img = im.bit1628(tmp_img)
        new_img = im.bit1628(new_img)
    tmp_img, new_img = tmp_img.astype(np.uint8), new_img.astype(np.uint8)
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
    if np.sum(centroid) != 0:
        warp = get_warp(rotation=warp, trans=centroid, size=size)
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


def imgs_transformECC_ver2(calc_img, centroids=False, motionType=1):
    # 移動補正をまとめた．
    warps = np.zeros((calc_img.shape[0] - 1, 2, 3), dtype=np.float32)
    warps[:, 0, 0], warps[:, 1, 1] = 1, 1
    if centroids is False:
        centroids = np.zeros((calc_img.shape[0]))
    else:
        centroids = np.modf(centroids)[0]  # 少数部分のみ
    tmp = np.max(calc_img, axis=(1, 2))
    roop = np.where((tmp[:-1]*tmp[1:]) != 0)[0]
    try_r = 0
    tmp_img = calc_img[0]
    for i in roop:  # 全部黒ならする必要ない．
        if try_r == 1:  # 直前の画像が移動補正に失敗した場合
            calc_img[i + 1][np.nonzero(calc_img[i + 1])] = np.max(calc_img[i])
            tmp_img = calc_img[i]
        calc_img[i + 1], warps[i], try_r = img_transformECC(tmp_img, calc_img[i + 1], motionType=motionType, centroid=centroids[i])
    return calc_img, warps, roop


def img_transformPOC(tmp_img, new_img):
    if np.max(tmp_img) > 255 or np.max(new_img) > 255:
        tmp_img = im.bit1628(tmp_img)
        new_img = im.bit1628(new_img)
    out, warp, _ = img_transformECC(tmp_img, new_img, motionType=1, centroid=False)
    tmp_img, out = tmp_img.astype(np.float32), out.astype(np.float32)
    tmp_img[tmp_img != 0] = np.max(out)
    size = new_img.shape    # 出力画像のサイズを指定
    # 移動を計算
    try:  # エラーでたら…
        d, etc = cv2.phaseCorrelate(tmp_img, out)  # d
        # warp = get_warp(d, size=size)  # 平行移動のみ
        m = np.float32([[1, 0, d[0]], [0, 1, d[1]]])
        # warp = add_warp(warp=warp, trans=d, size=size)  # 回転含む`
    except:
        print('平行移動に失敗した画像があります')
        warp = np.eye(2, 3, dtype=np.float32)
    # 元画像に対して，決めた変換メソッドで変換を実行
    out = cv2.warpAffine(out, m, size, flags=cv2.INTER_NEAREST)
    return out, warp


def imgs_transformPOC(calc_img):
    # 位相限定相関法．　回転不変位相限定相関法にしたい．平行移動のみ．
    calc_img_float = np.float32(calc_img)
    # フロンドのある画像のみ抽出
    tmp = np.max(calc_img, axis=(1, 2))
    roop = np.where((tmp[:-1] * tmp[1:]) != 0)[0]
    # 箱を作ったりの初期設定 
    warps = np.zeros((calc_img.shape[0] - 1, 2, 3), dtype=np.float32)
    warps[:, 0, 0], warps[:, 1, 1] = 1, 1
    size = calc_img[0].shape
    for i in roop:
        calc_img[i + 1], warps[i] = img_transformPOC(calc_img[i], calc_img[i + 1])
    return calc_img, warps, roop


def img_transform(warps, roops, *imgs):
    if roops is False:
        roops = range(warps.shape[0])
    for i in range(len(imgs)):
        size = imgs[i][0].shape
        for j in roops:
            imgs[i][j + 1] = cv2.warpAffine(imgs[i][j + 1], warps[j], size, flags=cv2.INTER_NEAREST)  # 平行移動の実行．
    return imgs


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
            ############## POC ####################
            print('POC')
            move_img, warps, roops = imgs_transformPOC(calc_img)
            trance.at[i, 'poc_move'] =  np.sum((move_img[0]!=0) != (move_img[-1]!=0))/(np.sum(move_img[-1] != 0))*0.5
            ############## ECCと重心 #################
            print('ECCと重')
            move_img, warps, roops = imgs_transformECC(calc_img, centroids=centroids)
            trance.at[i, 'cg_move'] =  np.sum((move_img[0]!=0) != (move_img[-1]!=0))/(np.sum(move_img[-1] != 0))*0.5

            ############### ECCのみ ################
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
