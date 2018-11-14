import numpy as np
# import matplotlib.pyplot as plt
import os
import sys
import glob
import datetime
import image_analysis as im
import pandas as pd


def lum2photon(data, dark=0):
    data[data < dark] = dark
    data[data > 0] = data[data > 0] - dark
    data[data < 0] = 1
    data_photon = data * 5.8 / (1 * 1200 * 0.9 / 100) / 240
    return data_photon

# pandasを用いてCSVファイルからデータを取り込み，ピークを出力するvar


def img2photon(folder, dark=0, dT=60, offset=0, save=True):
    print(folder)
    folder_list = sorted(os.listdir(folder))
    day = os.path.split(folder)[0]
    today = datetime.datetime.today().strftime("%Y-%m-%d")
    img_number = len(glob.glob(os.path.join(folder, folder_list[0], 'mask_frond_lum', '*.tif')))
    time_list = np.linspace(0, (img_number - 1) * dT / 60, img_number, dtype=np.float64) + offset
    frond_lum = np.empty((img_number, len(folder_list)), dtype=np.uint32)
    frond_area = np.empty((img_number, len(folder_list)), dtype=np.uint16)
    frond_avg = np.empty_like(frond_area)
    for (i, s) in enumerate(folder_list):
        # print(s)
        img = im.read_imgs(os.path.join(folder, s, 'mask_frond_lum')).astype(np.float64)
        img = lum2photon(img, dark=dark)
        frond_lum[:, i] = np.sum(img, axis=(1, 2))
        frond_area[:, i] = np.count_nonzero(img, axis=(1, 2))
    frond_avg = frond_lum / frond_area

    frond_lum, frond_avg, frond_area = pd.DataFrame(frond_lum, columns=folder_list, index=time_list), pd.DataFrame(
        frond_avg, columns=folder_list, index=time_list), pd.DataFrame(frond_area, columns=folder_list, index=time_list)
    frond_lum, frond_avg, frond_area = frond_lum.replace(0, np.nan), frond_avg.replace(0, np.nan), frond_area.replace(0, np.nan)
    # frond_lum.index, frond_area.index, frond_avg.index = time_list, time_list, time_list
    if os.path.exists(os.path.join(day, 'result')) is False:
        os.mkdir(os.path.join(day, 'result'))
    if save is True:
        frond_lum.to_csv(os.path.join(day, 'result', today + '-frond_photo_sum.csv'))
        frond_avg.to_csv(os.path.join(day, 'result', today + '-frond_photo_avg.csv'))
        frond_area.to_csv(os.path.join(day, 'result', today + '-frond_area.csv'))
    return frond_lum, frond_avg, frond_area


if __name__ == '__main__':
    # os.chdir(os.path.join("/Users", "kenya", "keisan", "python", "00data"))
    os.chdir(os.path.join("/hdd1", "kenya", "Labo", "keisan", "python", "00data"))
    days = "./170829-LL2LL-ito-MVX"
    dT = 60
    offset = 0
    dark = 1563.0
    frond_folder = os.path.join(days, "frond")
    img2photon(frond_folder, dT=dT, offset=offset, dark=dark, save=True)

    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dT = 60
    offset = 1.5
    dark = 1565.8
    frond_folder = os.path.join(days, "frond")
    img2photon(frond_folder, dT=dT, offset=offset, dark=dark, save=True)

    #############################################
    days = "./170215-LL2LL-MVX"
    dT = 60 + 10 / 60
    offset = 0
    dark = 1578.3
    frond_folder = os.path.join(days, "frond")
    img2photon(frond_folder, dT=dT, offset=offset, dark=dark, save=True)

    sys.exit()
    ############################################
    ################ 総発光量 ##################
    ############################################
    data_file = "2018-01-07frond_lum_sum_label.csv"
    avg = False
    ymax = 20000000
    loc = "out right"

    days = "./170829-LL2LL-ito-MVX"
    dT = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0, loc=loc)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dT = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0, loc=loc)
    #############################################
    days = "./170215-LL2LL-MVX"
    dT = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0, loc=loc)

    ############################################
    ################# 面積 #####################
    ############################################
    data_file = "2018-01-07frond_area_label.csv"
    avg = False
    ymax = 6000
    loc = "out right"

    days = "./170829-LL2LL-ito-MVX"
    dT = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0, loc=loc)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dT = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0, loc=loc)
    #############################################
    days = "./170215-LL2LL-MVX"
    dT = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0, loc=loc)

    ############################################
    ################# 面積 #####################
    ############################################

    data_file = "2018-01-07frond_area_label.csv"
    avg = False
    ymax = 6000

    days = "./170829-LL2LL-ito-MVX"
    dT = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dT = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0)
    #############################################
    days = "./170215-LL2LL-MVX"
    dT = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dT=dT, ymax=ymax, offset=0)

    ################################################
    ################# 総発光量 #####################
    ################################################
