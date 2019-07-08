# -*- coding: utf-8 -*-
"""Function to correct front movement."""

import glob
import os
import sys
import tkinter as tk

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import cv2
import image_analysis as im
import numpy as np
import pandas as pd
from PIL import Image, ImageTk


class CheckRotationGui:
    """回転確認のダイアルボックスを出す"""

    def __init__(self, img, angle):
        """a."""
        self.angle = angle
        self.img = img
        self.a = ''

    def radiobutton_box(self):
        """a."""
        self.tki = tk.Tk()
        self.tki.geometry()
        self.tki.title("フロンド回転の確認")
        img = Image.fromarray(self.img, "L")
        img = ImageTk.PhotoImage(master=self.tki, image=img)
        print(img)
        tk.Label(self.tki, image=img, text=str(self.angle) + "度回転しました．-90〜90度の回転になります．", compound="top").pack(side="top")
        var = tk.IntVar()  # チェックの有無変数
        rdo_box = ["これで良い", "これで良い", "回転角度を指定する"]
        var.set(0)
        for i in range(len(rdo_box)):
            tk.Radiobutton(self.tki, value=i, variable=var, text=rdo_box[i]).pack(side="top", anchor="w")
        tk.Label(self.tki, text="回転させる角度").pack(side="left", anchor="w")
        txt = tk.Entry(self.tki, width=20)
        txt.pack(side="left", anchor="w")

        def ok_btn():
            self.a = var.get()
            if self.a == 2:
                self.angle = int(txt.get())
            self.tki.quit()
            self.tki.destroy()

        tk.Button(self.tki, text='OK', command=ok_btn).pack(side="bottom")
        self.tki.mainloop()
        return self.a, self.angle


def img_transformECC(tmp_img, new_img, motionType=1, warp=np.eye(2, 3, dtype=np.float32)):
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
    tmp_img_8, new_img_8 = tmp_img.astype(np.uint8), new_img.astype(np.uint8)
    # 上記に合わせた，WarpMatrixを指定　warpは結果を格納するところ
    size = new_img.shape    # 出力画像のサイズを指定
    # 移動を計算
    warp = warp.astype(np.float32)
    try:  # エラーでたら…
        #     # (cc, warp_matrix) = cv2.findTransformECC(new_img_8, tmp_img_8, warp, motionType=motionType, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.1))
        cc, warp_matrix = cv2.findTransformECC(new_img_8, tmp_img_8, warp, motionType=motionType, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.5))
        temp = 0
    except:
        print('移動に失敗した画像があります')
        temp = 1
        warp_matrix = warp
    # 元画像に対して，決めた変換メソッドで変換を実行
    out = cv2.warpAffine(new_img_8, warp_matrix, size, flags=cv2.INTER_NEAREST)
    temp = 0
    # out[np.nonzero(out)] = np.max(out) - temp
    return out, warp, temp


def imgs_transformECC_ver2(calc_imgs, motionType=1, align=True):
    """Align the direction of frond. The movement is corrected based on the latest images.

    Args:
        calc_imgs: [description]
        motionType: [0: tanslation, 1:Euclidean, 2:affine, 3:homography] (default: {1})

    Returns:
        [description]
        [type]
    """
    # 移動補正をまとめた．
    warps = np.zeros((calc_imgs.shape[0], 2, 3), dtype=np.float64)
    warps[:, 0, 0], warps[:, 1, 1] = 1, 1
    tmp = np.sum(calc_imgs, axis=(1, 2))
    # roop = np.where(tmp != 0)[0]
    roop = np.where((tmp[:-1] * tmp[1:]) != 0)[0]
    tmp_idx = np.argmax(tmp)

    def rotation_reference_image(img):
        """Specify the direction of the frond after rotation."""
        img = img.astype(np.uint8)
        imgEdge, contours, hierarchy = cv2.findContours(img, 1, 2)
        w, h, angle = cv2.fitEllipse(contours[0])  # 楕円に近似
        center = tuple((np.array(img.shape) / 2).astype(np.uint16))

        def rotation_by_angle(img, center, angle):
            warp = cv2.getRotationMatrix2D(center, angle, 1)  # 回転行列を求める
            out = cv2.warpAffine(img, warp, img.shape, flags=cv2.INTER_NEAREST)
            check = CheckRotationGui(out, angle)
            a, angle = check.radiobutton_box()
            if a == 1:
                warp = cv2.getRotationMatrix2D(center, angle + 180, 1)  # 回転行列を求める
                out = cv2.warpAffine(img, warp, img.shape, flags=cv2.INTER_NEAREST)
            return out, a, angle, warp
        a = 2
        while a == 2:
            out, a, angle, warp = rotation_by_angle(img, center, angle)
        if ((angle + 90) / 180) % 2 == 1:
            out = out[::-1, ::-1]
            warp = cv2.getRotationMatrix2D(center, angle + 180, 1)
        return out, warp

    if align is True:
        calc_imgs[tmp_idx], tmp_warp = rotation_reference_image(calc_imgs[tmp_idx])
        warps[tmp_idx] = np.copy(tmp_warp)
    for i in range(0, tmp_idx)[::-1]:  # 全部黒ならする必要ない．
        if i in roop:
            # calc_imgs[i + 1][np.nonzero(calc_imgs[i + 1])] = np.max(calc_imgs[i])
            calc_imgs[i], warps[i], _ = img_transformECC(calc_imgs[i + 1], calc_imgs[i], motionType=motionType, warp=tmp_warp)
            tmp_warp = np.copy(warps[i])
    tmp_warp = np.copy(warps[tmp_idx])
    for i in range(tmp_idx, calc_imgs.shape[0] - 1):
        if i in roop:
            calc_imgs[i + 1][np.nonzero(calc_imgs[i + 1])] = np.max(calc_imgs[i])
            calc_imgs[i + 1], warps[i + 1], _ = img_transformECC(calc_imgs[i], calc_imgs[i + 1], motionType=motionType, warp=tmp_warp)
            tmp_warp = np.copy(warps[i + 1])
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
    tmp = np.sum(move_imgs, axis=(1, 2))
    roop = np.where(tmp != 0)[0]
    for i in roop:  # 全部黒ならする必要ない．
        move_imgs[i] = cv2.warpAffine(move_imgs[i], warps[i], move_imgs.shape[1:], flags=cv2.INTER_NEAREST)
    return move_imgs


def frond_transform(parent_directory, calc_folder="mask_frond", other_folder_list=["mask_frond_lum", 'frond'], motionType=1, align=True):
    """移動補正を一括化

    Args:
        parent_directory: [description]
        calc_folder: [description] (default: {"mask_frond"})
        other_folder_list: [description] (default: {["mask_frond_lum", 'frond']})
        motionType: [description] (default: {1})
    """
    calc_imgs = im.read_imgs(os.path.join(parent_directory, calc_folder))
    move_img, warps, roops = imgs_transformECC_ver2(calc_imgs, motionType=motionType, align=align)
    im.save_imgs(os.path.join(parent_directory, "moved_" + calc_folder), move_img)
    for i in other_folder_list:
        other_imgs = im.read_imgs(os.path.join(parent_directory, i))
        other_imgs = imgs_transformECC_warp(other_imgs, warps=warps)
        im.save_imgs(os.path.join(parent_directory, "moved_" + i), other_imgs)
    np.save(os.path.join(parent_directory, 'warps.npy'), warps)

if __name__ == '__main__':
    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data"))
    # 処理したいデータのフォルダ
    days = (['./170215-LL2LL-MVX'])
    # days = (['./170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX'])
    # days = (['./170829-LL2LL-ito-MVX'])
    for day in days:
        frond_folder = day + '/frond'
        folder_list = sorted(glob.glob(frond_folder + '/*'))
        trance = pd.DataFrame(index=[], columns=['first'])
        for i in folder_list:
            frond_transform(parent_directory=i, calc_folder="mask_frond", other_folder_list=["mask_frond_lum", 'frond'], motionType=1)
        trance.to_csv(os.path.join(day, 'tranc.csv'))
