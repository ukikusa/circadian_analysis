import numpy as np
import os
import itertools
import glob
import sys
import pandas as pd
import image_analysis as im
import cv2
from make_phase_img import img_to_mesh_phase
from frond_traker import frond_traker
from label2folder import label2frond


if __name__ == '__main__':
    os.chdir(os.path.join('/hdd1', 'kenya', 'Labo', 'keisan', 'python', '00data', "160728_UBQ"))
    ##################################################################
    ################### ここからUBQの前処理 ##########################
    ##################################################################
    folder = "data_light"
    dT = 30

    # ラベル2の解析作業
    background = 5196  # 手動で50×50ピクセル
