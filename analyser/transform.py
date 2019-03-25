# -*- coding: utf-8 -*-
"""Function to correct front movement."""

import glob
import os

import cv2
import image_analysis as im
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def img_transformECC(tmp_img, new_img, motionType=1):
    """Estimate rotation matrix by transformECC.

    Args:
        tmp_img: Template image
        new_img: Moving image
        motionType: [0: tanslation, 1:Euclidean, 2:affine, 3:homography] (default: {1})
        centroid: [description] (default: {[0,0]})

    Returns:
        [description]
        [type]
    """
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
    """移動補正をまとめた．"""
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
    """Motion correction based on a specific image.

    Args:
        calc_imgs: [description]
        motionType: [0: tanslation, 1:Euclidean, 2:affine, 3:homography] (default: {1})

    Returns:
        [description]
        [type]
    """
    # 移動補正をまとめた．
    warps = np.zeros((calc_imgs.shape[0], 2, 3), dtype=np.float32)
    warps[:, 0, 0], warps[:, 1, 1] = 1, 1
    tmp = np.sum(calc_imgs, axis=(1, 2))
    roop = np.where(tmp != 0)[0]
    tmp_idx = np.argmax(tmp)

    def rotation_reference_image(img):
        """Specify the direction of the frond after rotation."""
        img = img.astype(np.uint8)
        imgEdge, contours, hierarchy = cv2.findContours(img, 1, 2)
        w, h, angle = cv2.fitEllipse(contours[0])  # 楕円に近似
        center = tuple((np.array(img.shape) / 2).astype(np.uint8))

        def rotation_by_angle(img, center, angle):
            warp = cv2.getRotationMatrix2D(center, angle, 1)  # 回転行列を求める
            out = cv2.warpAffine(img, warp, img.shape, flags=cv2.INTER_NEAREST)
            print("\n---------------------------\n" + str(angle) + "度回転しました．\n良ければ0\n上下を反転させたければ1\n自分で回転角度を指定する場合は2を入力してください．")
            plt.imshow(out)
            plt.gray()
            plt.ion()
            plt.show()
            a = int(input())
            if a == 1:
                warp = cv2.getRotationMatrix2D(center, angle + 180, 1)  # 回転行列を求める
                out = cv2.warpAffine(img, warp, img.shape, flags=cv2.INTER_NEAREST)
            return out, a

        out, a = rotation_by_angle(img, center, angle)
        while a == 2:
            print("回転したい角度を入力してください．\n")
            angle = float(input())
            plt.close()
            out, a = rotation_by_angle(img, center, angle)
        plt.close()
        return out

    tmp_img = rotation_reference_image(calc_imgs[tmp_idx])
    try_r = 0
    for i in roop:  # 全部黒ならする必要ない．
        if try_r == 1:  # 直前の画像が移動補正に失敗した場合
            calc_imgs[i][np.nonzero(calc_imgs[i])] = np.max(calc_imgs[i - 1])
            tmp_img[np.nonzero(tmp_img)] = calc_imgs[i - 1]
        calc_imgs[i], warps[i], try_r = img_transformECC(tmp_img, calc_imgs[i], motionType=motionType)
    return calc_imgs, warps, roop


def imgs_transformECC_warp(move_imgs, warps):
    """Function to move images from the movement matrix.

    Args:
        move_imgs:
        warps: _, warps, _ = imgs_transformECC_ver2
        motionType: [0: tanslation, 1:Euclidean, 2:affine, 3:homography] (default: {1})

    Returns:
        move_imgs:
    """
    tmp = np.sum(calc_imgs, axis=(1, 2))
    roop = np.where(tmp != 0)[0]
    for i in roop:  # 全部黒ならする必要ない．
        move_imgs[i] = cv2.warpAffine(move_imgs[i], warps[i], move_imgs.shape[1:], flags=cv2.INTER_NEAREST)
    return move_imgs


def imgs_transformECC_all(calc_imgs, motionType=1, *other_imgs):
    calc_imgs, warps, _ = imgs_transformECC_ver2(calc_imgs, motionType=motionType)
    for i in range(len(other_imgs)):
        other_imgs[i] = imgs_transformECC_warp(other_imgs[i], motionType=motionType)
    return calc_imgs, warps, other_imgs

# To do
# warps = cv2.invertAffineTransform(warps) で逆行列が求まるのでもとに戻せる．

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
            calc_imgs = im.read_imgs(i + '/mask_frond')
            mask_lum_img = im.read_imgs(i + '/mask_frond_lum')
            lum_img = im.read_imgs(i + '/frond_lum')
            trance.at[i, 'first'] = np.sum(calc_imgs[0] != 0)
            trance.at[i, 'last'] = np.sum(calc_imgs[-1] != 0)
            imgs_transformECC_all(calc_imgs, 1)
            move_img, warps, roops = imgs_transformECC_ver2(calc_imgs)
            trance.at[i, 'ECC_moved_2'] = np.sum((move_img[0] != 0) != (move_img[-1] != 0)) / (np.sum(move_img[-1] != 0)) * 0.5
            # np.save(i + '/warps.npy', warps)
            # warps.resize(((calc_img.shape[0] - 1), 3))
            # np.savetxt(i + '/warps.csv', warps, delimiter=',')
            # im.save_imgs(i + '/moved_mask_frond', move_img)
            # im.save_imgs(i + '/moved_mask_frond_lum', imgs[0])
            # im.save_imgs(i + '/moved_frond_lum', imgs[1])
        trance.to_csv(os.path.join(day, 'tranc.csv'))
