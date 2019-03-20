# -*- coding: utf-8 -*-
"""Image analysis."""

import glob
import itertools
import os
import sys

import cv2
import numpy as np
from PIL import Image  # Pillowの方を入れる．PILとは共存しない


def read_imgs(img_folder, color=False, extension='tif'):  # 画像入っているフォルダ
    file_list = sorted(glob.glob(os.path.join(img_folder, '*.' + extension)))
    if len(file_list) == 0:
        print(img_folder + '/*.' + extension + 'がありません')
        sys.exit()
    if color is False:  # グレースケール読むとき
        imread_type = cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH
    elif color is True:  # カラー読むとき
        imread_type = cv2.IMREAD_COLOR
    tmp = cv2.imread(file_list[0], imread_type)  # 1枚目
    img = np.empty(np.concatenate(([len(file_list)], tmp.shape)), dtype=tmp.dtype)  # 箱
    img[0] = tmp  # 1枚目を箱に
    for i in range(1, len(file_list)):  # 2枚目から全部取り込み
        img[i] = cv2.imread(file_list[i], imread_type)
    print(img_folder + 'から' + str(i + 1) + '枚取り込みました．')
    return img


def save_imgs(save_folder, img, file_name='', extension='tif', idx='ALL'):
    # 画像スタックになっている配列から，save_folderに画像を保存．
    if idx == 'ALL':
        idx = np.arange(img.shape[0])
    if idx.size == 0:
        print(save_folder + 'に保存する画像がありません')
        return 0
    if os.path.exists(save_folder) is False:
        os.makedirs(save_folder)
    if extension == 'png':  # pngなんだから非圧縮で保存しようよ
        for i in idx:
            Image.fromarray(img[i]).save(os.path.join(save_folder, file_name + str(i).zfill(3) + '.' + extension))
    else:  # tifはデフォルト非圧縮 jpgは圧縮される．
        for i in idx:
            Image.fromarray(img[i]).save(os.path.join(save_folder, file_name + str(i).zfill(3) + '.' + extension))
    print(str(save_folder) + 'に保存しました')


def bit1628(img):  # 16bit 画像を8bit画像に
    if np.max(img) > 255 or np.min(img) >= 10:
        img8 = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    return img8


def merge_imgs(imgs, imgs_merge):  # mergeする
    imgs_merge = np.not_equal(imgs_merge, 0)
    if imgs.ndim == 4:
        imgs_merge = imgs_merge.repeat(3)
        imgs_merge = np.reshape(imgs_merge, imgs.shape)
        print(imgs_merge.shape)
    imgs_out = imgs * imgs_merge
    return imgs_out


def make_color(phase, grey=-2, black=-1):
    # phaseのデータをRGB配列に．
    # 画像の格納庫
    hsv = np.ones(np.concatenate((phase.shape, [3])), dtype=np.uint8) * 255
    hsv[::, ::, 0] = (phase * 180).astype(np.uint8)
    black = np.where(hsv[:, :, 0] == -180)
    hsv[np.isnan(phase), :] = [0, 0, 0]
    if black is not False:
        hsv[phase == black, :] = [0, 0, 0]
    if grey is not False:
        hsv[phase == grey, :] = [165, 2, 69]  # グレーに
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[:, :, ::-1]  # HSV→BGR→RGB
    return rgb


def make_colors(img, grey=-2, black=-1):
    color = np.empty((np.concatenate((img.shape, [3]))), dtype=np.uint8)
    for i in range(img.shape[0]):
        color[i] = make_color(img[i], grey=grey, black=black)
    return color


def mesh_imgs(folder, mesh=5):
    """Make the image mash by taking the average in the vicinity."""
    # 画像のフォルダから画像を全部読み込んできて，全てメッシュ化してしまおう！
    img = read_imgs(folder)
    # 以下メッシュ化するよ！
    meshed = np.empty((img.shape[0], int(img.shape[1] / mesh), int(img.shape[2] / mesh)))
    for i, j in itertools.product(np.arange(int(img.shape[1] / mesh)), np.arange(int(img.shape[2] / mesh))):
        meshed[::, i, j] = img[::, i * mesh:(i + 1) * mesh, j * mesh:(j + 1) * mesh].mean(axis=(1, 2))
    return meshed


def past_img(img, img2, margin=0, dtype=np.uint16, folder=False):
    # imgを横に並べる．間はmarginuピクセル カラー非対応．
    if folder != 0:  # 読み込み
        img, img2 = read_imgs(img), read_imgs(img2)
    if img.ndim == 2:  # 一枚だけの画像のとき
        new_img = np.empty((img.shape[0], img.shape[1] + margin + imgs.shape[1]), dtype=dtype)
        new_img[:, :img.shape[1]] = img
        new_img[:, -img2.shape[1]:] = img2
    else:
        new_img = np.empty((img.shape[0], img.shape[1], img.shape[2] + margin + img.shape[2]), dtype=dtype)
        new_img[:, :, :img.shape[2]] = img
        new_img[:, :, -img2.shape[2]:] = img2
    return new_img


if __name__ == "__main__":
    imgs = np.zeros((3, 100)).astype(np.float64)
    imgs[0, :] = np.arange(0, 1, 0.01)
    imgs[1, :] = np.arange(0, 1, 0.01)
    imgs[2, :] = np.arange(0, 1, 0.01)
    color = make_color(imgs)
    print(color)
    cv2.imwrite('color.tif', color)
