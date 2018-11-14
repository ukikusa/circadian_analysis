# -*-coding: utf-8 -*-

import numpy as np
import os
import glob
import sys
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import image_analysis as im
import peak_analysis as pa


if __name__ == '__main__':
    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    days = (['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX'])
    # 出力
    period_folder = '/hdd1/kenya/Labo/keisan/python/_171019/result'
    for day in days:
        frond_folder = os.path.join(period_folder, day)
        for i in sorted(glob.glob(os.path.join(frond_folder, '*'))):
            print(i)
            fft_folder = i + '22_94/Ampnorm_dataFFTnlls_data_0_71'
            phase_rae = np.loadtxt(fft_folder + '/FFTnlls_results.csv', skiprows=1)

            phase_avg = np.average(phase_rae[:, 2])
            # とりあえず座標の設定と画像の整形
            save_folder = + '/result/' + day.split('/')[1] + '/' + os.path.split(i)[1]
            if os.path.exists(save_folder) == False:
                os.makedirs(save_folder)
            name = np.array([x, y, pixel_linalg])
            name = np.array(name)
            data_AmpNorm = AmpNorm(datas, w=24, dT=dT, save_folder=save_folder, pdf=False, name=name)
            data_cut = cut_time_data(datas, x, s_point=10, e_point=50)

            # for_fft = np.arange(0, 24*3 + int(out.shape[0]/24-5)*24, 24)
            # os.mkdir(os.path.join(i, 'result_171016'))
            # for j in for_fft:
            # out_tmp = out[j+2:j+24*3+2]
            # cut_time_data(out, frond_x, frond_y, )
            # if np.sum(np.all(out_tmp != 0, axis=0)) != 0:
            #     if out_little != []:
            #         np.savetxt(os.path.join(i, 'result_171016', str(j) + 'out.csv'), out_little[:, out_little[2]>21], delimiter=',')
