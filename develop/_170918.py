# -*- coding: utf-8 -*-

import numpy as np
import os
import cv2
import glob
# import sys
# import math
# import datetime
# import image_analysis as im
# import make_phase_img as mpi
# import frond_traker
import PyPDF2

if __name__ == '__main__':
    os.chdir('/hdd1/kenya/Labo/keisan/python/00data')
    # os.chdir('/home/kenya/Dropbox/Labo/python/00data')
    # 処理したいデータのフォルダ
    day = ('./170215-LL2LL-MVX')
    # 解析データのフォルダ
    pdf_folder = os.path.join(day, 'result', 'R')
    pdf = sorted(glob.glob(os.path.join(pdf_folder, '*.pdf')))
    for i in pdf:
        PyPDF2.PdfFileMerger.append(i)
    PyPDF2.write(os.path.join(os.path.split(pdf_folder)[0], '_R.pdf'))
    PyPDF2.PdfFileMerger.close()


    # frond_traker(imgs, min_size=20)
    # mpi.img_to_mesh_phase(folder, avg, mesh=3, dT=60, peak_avg=3, p_range=12, fit_range=5, save=True)
