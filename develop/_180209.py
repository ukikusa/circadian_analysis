import numpy as np
import os
import itertools
import glob
import sys
import pandas as pd
import image_analysis as im
import cv2
from _171214_local import threshold_imgs
from make_phase_img import img_to_mesh_phase


if 0:
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'MVX', '160715'))
    data_folder = os.path.join('.', 'data_light')
    imgs = im.read_imgs(data_folder, color=False)
    local, gaussian = False, 3
    imgs_thresh = threshold_imgs(imgs, thresh=cv2.THRESH_OTSU, gaussian=gaussian, local=local, kernel=10)[0]
    img_thresh = np.max(imgs_thresh[0:63], axis=0)
    im.save_imgs(os.path.join('thresh' + '_gaussian-' + str(gaussian)), imgs_thresh)
    print(img_thresh)
    cv2.imwrite('thresh.tif', img_thresh)


if 0:
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'MVX', '160715'))
    img_lum = im.read_imgs('data', color=False)
    img_frond = cv2.imread('thresh_edit.tif', cv2.IMREAD_GRAYSCALE)
    frond_lum = np.zeros_like(img_lum)
    print(img_frond == 0)
    frond_lum[:, img_frond == 0] = img_lum[:, img_frond == 0]
    print(np.average(frond_lum[frond_lum != 0]))
    frond_lum[frond_lum > 10000] = np.average(frond_lum[frond_lum != 0])
    im.save_imgs('frond_lum', frond_lum)

if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'MVX', '160715'))
    dT = 30
    color, imgs_phase = img_to_mesh_phase('frond_lum', avg=3, mesh=1, dT=dT, peak_avg=3, p_range=24, fit_range=7, save_folder='color')
