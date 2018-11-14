import numpy as np
import os
# import matplotlib.pyplot as plt
import image_analysis as im
# import peak_analysis as pa


def blanch_cut_byfor(img_dst):
    # img_dst = np.random.randint(0, 2, (512, 512))
    blanch_number = 1
    img_dst[(0, -1), :] = 0
    img_dst[:, (0, -1)] = 0
    while blanch_number != 0:
        blanch_number = 0
        index = np.where(img_dst == 1)
        for i in np.arange(len(index[0])):
            if np.sum(img_dst[index[0][i] - 1:index[0][i] + 2, index[1][i] - 1:index[1][i] + 2]) <= 2:
                img_dst[index[0][i], index[1][i]] = 0
                blanch_number = blanch_number + 1
        if blanch_number == 0:
            break
        else:
            blanch_number = 0
            index = np.where(img_dst == 1)
        for i in np.arange(len(index[0]))[::-1]:
            if np.sum(img_dst[index[0][i] - 1:index[0][i] + 2, index[1][i] - 1:index[1][i] + 2]) <= 2:
                img_dst[index[0][i], index[1][i]] = 0
                blanch_number = blanch_number + 1
    return img_dst


def blanch_cat_bynp(imgs):  # 画像に一括で処理
    imgs_tmp = np.zeros_like(imgs)
    imgs[:, (0, -1), :], imgs[:, :, (0, -1)] = 0, 0  # 周辺部を削除
    imgs_old = imgs  # 処理前の画像
    while np.sum(imgs_old != imgs_tmp):
        imgs_tmp[:, 1:-1, 1:-1] = imgs[:, :-2, :-2] + imgs[:, :-2, 1, -1] + imgs[:, :-2, 2:] + imgs[:, 1:-1, :-2] + \
            imgs[:, 1:-1, 1, -1] + imgs[:, 1:-1, 2:] + imgs[:, 2:, :-2] + imgs[:, 2:, 1, -1] + imgs[:, 2:, 2:]
        imgs_tmp[np.logical_or(imgs == 0, imgs_tmp <= 2)] = 0  # ブランチ削除
        imgs_tmp[imgs_tmp.nonzero] = 1
        imgs_old = imgs  # 評価用に処理に使った画像を残しておく
        imgs = imgs_tmp
    return imgs

if __name__ == '__main__':
    os.chdir(os.path.join("/Users", "kenya", "keisan", "python","00data"))
    img_folder = os.path.join(".","folder_name")
    save_folder = os.path.join(".","folder_name")
    
    imgs = im.read_imgs(img_folder)
    imgs = blanch_cat_bynp(imgs)
    im.save_imgs(save_folder,imgs)
