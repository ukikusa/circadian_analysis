# coding: utf-8
"""Fanction group for editing duckweed image as preprocessing of analysis."""
import numpy as np
import cv2


def label_img(img_edge, img, connectivity=4, inversion=False):
    """A function tha outputs an image labeled from the original image and the boundary image."""
    # ラベリングする．(inbersion: False;白 True;黒)を．
    img_edge = img_edge.astype(np.uint8)
    if inversion is True:
        img_edge = 255 - img_edge
    # ラベリングして，それぞれに中央値入れる！
    # connetorakkinnguctivity 斜めがつながってるか．4;つながってる, 8; つながってない．
    n, img_labels = cv2.connectedComponents(img_edge, connectivity=connectivity)
    result = np.zeros_like(img)
    for i in range(1, n):
        result[img_labels == i] = np.median(img[img_labels == i])
    return result


def label_imgs(imgs_edge, imgs, connectivity=4, inversion=False):
    """A function tha outputs an image group labeled from the original image group and the boundary image group."""
    # ラベリングする．(inbersion: False;白(255) True;黒(0))を．
    result = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        result[i, :, :] = label_img(imgs_edge[i, :, :], imgs[i, :, :], connectivity=connectivity, inversion=inversion)
    return result
