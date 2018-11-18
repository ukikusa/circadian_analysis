# -*-coding: utf-8 -*-
import skimage
from skimage import morphology
import cv2
import os
import glob
import numpy as np
import image_analysis as im  # 自作


def skeletonize_folder(folder, save_folder):
    imgs = im.read_imgs(folder)
    out = np.zeros_like(imgs)
    imgs_ = np.zeros_like(imgs, dtype=np.uint8)
    imgs_[imgs != 0] = 1  # 二値化が必須
    kernel = np.ones((3, 3))
    for i in range(imgs.shape[0]):
        tmp = morphology.dilation(imgs_[i], kernel) - imgs_[i]
        skeletonize = morphology.skeletonize(tmp).astype(np.uint8)
        print(skeletonize.shape)
        skeletonize = skeletonize + 255
        label_n, label_img = cv2.connectedComponents(skeletonize, connectivity=4)
        print(label_n)
        for j in range(1, label_n):
            print(np.sum(label_img == j))
            out[i][label_img == j] = np.argmax(np.bincount(imgs[i][label_img == j]))
    im.save_imgs(save_folder, out)
    return out


if __name__ == '__main__':
    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data"))
    # dataフォルダ
    days = (['170613-LD2LL-ito-MVX', './170215-LL2LL-MVX', './170829-LL2LL-ito-MVX'])
    for day in days:
        # 解析データのフォルダ
        label_folder = os.path.join(day, 'edit_raw', 'label_img')
        save_folder = os.path.join(day, 'edit_raw', 'label_img_skeleton')
        save_folder2 = os.path.join(day, 'edit_raw', 'dev_img_skeleton')
        # 出力先フォルダ
        out = skeletonize_folder(label_folder, save_folder)
        out[out != 0] = 255
        im.save_imgs(save_folder2, out)
