# coding: utf-8
import pandas as pd
import numpy as np
import image_analysis as im
import os
import itertools


if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    day = '170613-LD2LL-ito-MVX'
    # 子のフロンドの指定
    frond_phase_file = os.path.join(day, 'result', 'pdf', '2018-01-22-frond_photo_avgphase.csv')
    frond_phase = pd.read_csv(frond_phase_file, header=0, index_col=0)
    print(frond_phase.columns)
    frond_idx = input('子のインデックス(0-'+str(len(frond_phase.columns))+'): ')
    frond_idx = int(frond_idx)
    # 親のフロンドの指定
    img_list = sorted(os.listdir(os.path.join(day, 'frond')))
    print(img_list)
    img_idx = input('親のインデックス(0-'+str(len(img_list))+'): ')
    img_idx = int(img_idx)
    img_file = os.path.join(day, 'frond', img_list[img_idx], 'small_phase_mesh1_avg3.npy')
    img = np.load(img_file)
    # 子フロンドデータの抽出
    phase = np.array(frond_phase.values)[:, frond_idx]
    phase[phase == -1] = np.nan
    # 位相差作成
    diff_imgs = np.empty_like(img)
    for x, y in itertools.product(range(img.shape[1]), range(img.shape[2])):
        diff_imgs[:, x, y] = img[:, x, y] - phase
    diff_imgs[diff_imgs > 0.5] = diff_imgs[diff_imgs > 0.5]-1
    diff_imgs[diff_imgs < -0.5] = diff_imgs[diff_imgs < -0.5]+1
    color_diff = np.zeros(np.concatenate((diff_imgs.shape, [3])), dtype=np.uint8)
    # インデックス作成
    all_idx = ~np.isnan(diff_imgs)
    plus_idx = all_idx * diff_imgs >= 0
    minus_idx = all_idx * diff_imgs <= 0
    # 色作成
    print(np.nanmin(diff_imgs))
    color_diff[all_idx] = 255
    color_diff[plus_idx, 1] = (diff_imgs[plus_idx]*255*2).astype(np.uint8)
    color_diff[plus_idx, 2] = (diff_imgs[plus_idx]*255*2).astype(np.uint8)
    color_diff[minus_idx, 0] = (diff_imgs[minus_idx]*255*2).astype(np.uint8)
    color_diff[minus_idx, 1] = (diff_imgs[minus_idx]*255*2).astype(np.uint8)
    print(os.path.join(day, 'result', 'diff', str(img_list[img_idx])+str(img_list[frond_idx])))
    im.save_imgs(os.path.join(day, 'result', 'diff', str(img_list[img_idx])+'-'+str(img_list[frond_idx])), color_diff)
