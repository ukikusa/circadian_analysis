import numpy as np
import os
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
import pandas as pd
from make_phase_img import phase_fit
from _180113_lum2photo import lum2photon


def fit_fig(data, pdf_save, avg=3, peak_avg=3, p_range=6, fit_range=4):
    # if os.path.exists(pdf_save) is False:
    #     os.makedirs(pdf_save)
    time = np.arange(len(data), dtype=np.float64)
    time_move = time[int(avg / 2): data.shape[0] - int(avg / 2)]
    plt.figure(figsize=(7, 3), dpi=100)
    pa.peak_find(data, time=time_move, avg=avg, peak_avg=peak_avg, p_range=p_range, fit_range=fit_range, pdf_save=pdf_save)

    # plt.show()
    plt.savefig(os.path.join(pdf_save, "before_fit_fig.pdf"))
    plt.close()
    return 0


def frond_fit_fig(data, pdf_save, avg=3, peak_avg=3, p_range=6, fit_range=4):
    # if os.path.exists(pdf_save) is False:
    #     os.makedirs(pdf_save)
    time = np.arange(len(data), dtype=np.float64)
    time_move = time[int(avg / 2): data.shape[0] - int(avg / 2)]
    plt.figure(figsize=(7, 3), dpi=100)
    pa.peak_find(data, time=time_move, avg=avg, peak_avg=peak_avg, p_range=p_range, fit_range=fit_range, pdf_save=pdf_save)
    plt.ylim([0, 10])
    # plt.ylim(bottom=0)
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(right="off", top="off")
    # plt.xlabel('time'
    plt.xlim([0, 168])
    plt.xticks(np.arange(0, 168, 24))
    plt.vlines(np.arange(0, 168, 24), 0, 10, colors="k", linestyle="dotted", label="", lw=0.5)
    # plt.show()
    plt.savefig(os.path.join(pdf_save, "before_frond_fit_fig.pdf"))
    plt.close()
    return 0


if __name__ == '__main__':
    os.chdir(os.path.join("/hdd1", "kenya", "Labo", "keisan", "python", "00data"))
    fit_range = 6
    p_range = 9
    pdf_save = '.'
    dark = 1563.0
    # data = pd.read_csv('170829-LL2LL-ito-MVX/result/2018-01-22-frond_photo_avg.csv', index_col=0, dtype=np.float64)
    # data = im.read_imgs('170613-LD2LL-ito-MVX/frond/label-001_239-188_n214/small_moved_mask_frond_lum')
    data = im.read_imgs('170829-LL2LL-ito-MVX/frond/label-001_316-158_n200/small_moved_mask_frond_lum')
    data = lum2photon(data, dark=dark)
    # data_values = np.array(data.values)
    data_value = data[:, 90, 90]
    # fit_fig(data_value, pdf_save, avg=3, p_range=p_range, fit_range=fit_range)
    frond_fit_fig(data_value, pdf_save, avg=3, p_range=p_range, fit_range=fit_range)
