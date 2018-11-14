# coding: utf-8
import numpy as np
import image_analysis as im
import os
import sys


def gray_color(img):
    color = np.empty((img.shape[0], img.shape[1], img.shape[2], 3), dtype=np.uint8)
    color[:, :, :, 0] = im.bit1628(img)
    color[:, :, :, 1] = color[:, :, :, 0]
    color[:, :, :, 2] = color[:, :, :, 0]
    return color


def bit_1628_pix2pix(day):
    img = im.read_imgs(os.path.join(day, 'edit_raw', 'div_tmp_01_3')).astype(np.uint16)
    # img[img!=0] = 256*256-1
    img2 = im.read_imgs(os.path.join(day, 'raw_data', 'data_light'))
    img2_std = np.empty((img2.shape), dtype=np.uint8)
    for i in range(img2.shape[0]):
        img2_std[i] = ((img2[i] - np.min(img2[i])) / (np.max(img2[i]) - np.min(img2[i])) * 255).astype(np.uint8)
    # img_paset = im.past_img(img, img2, dtype=np.uint32)
    return im.past_img(img, img2_std, dtype=np.uint8)


if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    # day = '170215-LL2LL-MVX'
    day = '170613-LD2LL-ito-MVX'
    img_past = bit_1628_pix2pix(day)
    print(np.arange(0, 58, 2))
    img_past2 = img_past[np.arange(0, 59, 2)]
    im.save_imgs(os.path.join('pix2pix', 'pix2pix-tensorflow-master', 'datasets', '_20180731', 'gray_8bit_data'), img_past2, extension='png', file_name='_170613_')
    im.save_imgs(os.path.join('pix2pix', 'pix2pix-tensorflow-master', 'datasets', '_20180731', 'gray_8bit_data_all'), img_past, extension='png', file_name='_170613_')
    # im.save_imgs(os.path.join('pix2pix', 'pix2pix-tensorflow-master', 'datasets', '_20180620', 'gray_16bit_std_data_all'), img_paset_std, extension='png',file_name='_170215_' )
    sys.exit()
    day = '170613-LD2LL-ito-MVX'
    img_past = bit_1628_pix2pix(day)
    im.save_imgs(os.path.join('pix2pix', 'pix2pix-tensorflow-master', 'datasets', '_20180702', 'gray_8bit_data_all'), img_past, extension='png', file_name='_17613_')

    day = '170829-LL2LL-ito-MVX'
    # day = '170613-LD2LL-ito-MVX'
    img_past = bit_1628_pix2pix(day)
    im.save_imgs(os.path.join('pix2pix', 'pix2pix-tensorflow-master', 'datasets', '_20180702', 'gray_8bit_data_all'), img_past, extension='png', file_name='_17829_')
