# -*- coding: utf-8 -*-
"""Function to correct front movement."""


import os
import sys
import tkinter as tk

import cv2
import numpy as np

# import pandas as pd
from PIL import Image, ImageTk

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
import image_analysis as im


class CheckRotationGui:
    """回転確認のダイアルボックスを出す"""

    def __init__(self, img, angle):
        """a."""
        self.angle = angle
        self.img = img
        self.a = ""

    def radiobutton_box(self):
        """a."""
        self.tki = tk.Tk()
        self.tki.geometry()
        self.tki.title("フロンド回転の確認")
        img = Image.fromarray(self.img, "L")
        img = ImageTk.PhotoImage(master=self.tki, image=img)
        tk.Label(
            self.tki, image=img, text=str(self.angle) + "度回転しました.", compound="top"
        ).pack(side="top")
        var = tk.IntVar()  # チェックの有無変数
        rdo_box = ["これで良い", "上下反転をする", "回転角度を指定する"]
        var.set(0)
        for i in range(len(rdo_box)):
            tk.Radiobutton(self.tki, value=i, variable=var, text=rdo_box[i]).pack(
                side="top", anchor="w"
            )
        tk.Label(self.tki, text="回転させる角度").pack(side="left", anchor="w")
        txt = tk.Entry(self.tki, width=20)
        txt.pack(side="left", anchor="w")

        def ok_btn():
            self.a = var.get()
            if self.a == 1:
                self.angle = self.angle + 180
            elif self.a == 2:
                self.angle = int(txt.get())
            self.tki.quit()
            self.tki.destroy()

        tk.Button(self.tki, text="OK", command=ok_btn).pack(side="bottom")
        self.tki.mainloop()
        return self.a, self.angle


def img_transformECC(
    tmp_img, new_img, motionType=1, warp=np.eye(2, 3, dtype=np.float32)
):
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
    size = new_img.shape  # 出力画像のサイズを指定
    # 移動を計算
    warp = warp.astype(np.float32)
    for ecc_threshold in np.arange(0.001, 1, 0.005):
        try:  # エラーでたら…
            _cc, warp_matrix = cv2.findTransformECC(
                new_img_8,
                tmp_img_8,
                warp,
                motionType,
                criteria=(
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    50,
                    ecc_threshold,
                ),
            )
        except:
            print(str(ecc_threshold) + "でエラー")
        else:
            break
    # 元画像に対して，決めた変換メソッドで変換を実行
    out = cv2.warpAffine(new_img_8, warp_matrix, size, flags=cv2.INTER_NEAREST)
    temp = 0
    return out, warp, temp


def imgs_transformECC(calc_imgs, motionType=1, align=True):
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
    moved_imgs = np.zeros_like(calc_imgs)
    tmp = np.sum(calc_imgs, axis=(1, 2))
    roop = np.where(tmp != 0)[0]
    # roop = np.where((tmp[:-1] * tmp[1:]) != 0)[0]
    tmp_idx = np.argmax(tmp)

    def rotation_reference_image(img):
        """Specify the direction of the frond after rotation."""
        img = img.astype(np.uint8)
        _imgedge, contours, _hierarchy = cv2.findContours(img, 1, 2)
        _w, _h, angle = cv2.fitEllipse(contours[0])  # 楕円に近似
        center = tuple((np.array(img.shape) / 2).astype(np.uint16))
        a = 2
        while a != 0:
            if ((angle + 90) // 180) % 2 == 1:
                warp = cv2.getRotationMatrix2D(center, angle + 180, 1)
                turn_pi = True
                out = cv2.warpAffine(img, warp, img.shape, flags=cv2.INTER_NEAREST)
                out = out[::-1, ::-1]
            else:
                warp = cv2.getRotationMatrix2D(center, angle, 1)  # 回転行列を求める
                turn_pi = False
                out = cv2.warpAffine(img, warp, img.shape, flags=cv2.INTER_NEAREST)
            check = CheckRotationGui(out, angle)
            a, angle = check.radiobutton_box()
        return out, warp, turn_pi

    if align is True:  # 最大のフロンドを見やすい向きの整形する．
        moved_imgs[tmp_idx], tmp_warp, trun_pi = rotation_reference_image(
            calc_imgs[tmp_idx]
        )
        warps[tmp_idx] = np.copy(tmp_warp)
        if trun_pi:
            moved_imgs[tmp_idx] = moved_imgs[tmp_idx][::-1, ::-1]
    else:
        trun_pi = False

    for i in roop:
        moved_imgs[i], warps[i], _ = img_transformECC(
            moved_imgs[tmp_idx], calc_imgs[i], motionType, tmp_warp
        )
        tmp_warp = np.copy(warps[i])
    if trun_pi:
        moved_imgs = moved_imgs[:, ::-1, ::-1]
    # for i in range(0, tmp_idx)[::-1]:  # 全部黒ならする必要ない．
    #     if i in roop:
    #         # calc_imgs[i + 1][np.nonzero(calc_imgs[i + 1])] = np.max(calc_imgs[i])
    #         calc_imgs[i], warps[i], _ = img_transformECC(calc_imgs[i + 1], calc_imgs[i], motionType=motionType, warp=tmp_warp)
    #         tmp_warp = np.copy(warps[i])
    # tmp_warp = np.copy(warps[tmp_idx])
    # for i in range(tmp_idx, calc_imgs.shape[0] - 1):
    #     if i in roop:
    #         # calc_imgs[i + 1][np.nonzero(calc_imgs[i + 1])] = np.max(calc_imgs[i])
    #         calc_imgs[i + 1], warps[i + 1], _ = img_transformECC(calc_imgs[i], calc_imgs[i + 1], motionType=motionType, warp=tmp_warp)
    #         tmp_warp = np.copy(warps[i + 1])
    return moved_imgs, warps, roop, trun_pi


def imgs_transformECC_warp(move_imgs, warps, trun_pi=False):
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
        move_imgs[i] = cv2.warpAffine(
            move_imgs[i], warps[i], move_imgs.shape[1:], flags=cv2.INTER_NEAREST
        )
    if trun_pi:
        move_imgs = move_imgs[:, ::-1, ::-1]
    return move_imgs


def frond_transform(
    parent_directory,
    calc_folder="mask_frond.tif",
    other_folder_list=["mask_frond_lum.tif", "frond_lum.tif"],
    motionType=1,
    align=True,
):
    """移動補正を一括化

    Args:
        parent_directory: [description]
        calc_folder: [description] (default: {"mask_frond"})
        other_folder_list: [description] (default: {["mask_frond_lum", 'frond']})
        motionType: [description] (default: {1})
    """
    calc_imgs = im.read_imgs(os.path.join(parent_directory, calc_folder))
    move_img, warps, _roops, trun_pi = imgs_transformECC(
        calc_imgs, motionType=motionType, align=align
    )
    im.save_imgs(parent_directory, move_img, file_name="moved_" + calc_folder)
    for i in other_folder_list:
        other_imgs = im.read_imgs(os.path.join(parent_directory, i))
        other_imgs = imgs_transformECC_warp(other_imgs, warps, trun_pi)
        im.save_imgs(parent_directory, other_imgs, file_name="moved_" + i)
    np.save(os.path.join(parent_directory, "warps.npy"), warps)
    with open(os.path.join(parent_directory, "trun_pi"), mode="w") as f:
        f.write(str(trun_pi))


if __name__ == "__main__":
    os.chdir(
        os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data")
    )
    # 処理したいデータのフォルダ
    # days = (['./170215-LL2LL-MVX'])
    # # days = (['./170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX'])
    # # days = (['./170829-LL2LL-ito-MVX'])
    # for day in days:
    #     frond_folder = day + '/frond'
    #     folder_list = sorted(glob.glob(frond_folder + '/*'))
    #     trance = pd.DataFrame(index=[], columns=['first'])
    #     for i in folder_list:
    #         frond_transform(parent_directory=i, calc_folder="mask_frond", other_folder_list=["mask_frond_lum", 'frond'], motionType=1)
    #     trance.to_csv(os.path.join(day, 'tranc.csv'))
