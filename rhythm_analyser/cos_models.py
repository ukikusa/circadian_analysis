# -*- coding: utf-8 -*-
'''fft_nllsをする．標準化には不偏標準偏差を使う'''

import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
import pandas as pd
import scipy as sp
from scipy.optimize import curve_fit
np.set_printoptions(precision=5, floatmode='fixed', suppress=True)


def cos_model(time, amp, tau, pha):  # fittingのモデル
    return amp * np.cos(2 * np.pi * (time / tau) + pha)


def cos_model_1(time, amp0, tau0, pha0):
    fit = 0
    for i in range(1):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_2(time, amp0, tau0, pha0, amp1, tau1, pha1):
    fit = 0
    for i in range(2):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_3(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2):
    fit = 0
    for i in range(3):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_4(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3):
    fit = 0
    for i in range(4):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_5(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4):
    fit = 0
    for i in range(5):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_6(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5):
    fit = 0
    for i in range(6):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_7(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6):
    fit = 0
    for i in range(7):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_8(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7):
    fit = 0
    for i in range(8):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_9(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7, amp8, tau8, pha8):
    fit = 0
    for i in range(9):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_10(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7, amp8, tau8, pha8, amp9, tau9, pha9):
    fit = 0
    for i in range(10):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_11(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7, amp8, tau8, pha8, amp9, tau9, pha9, amp10, tau10, pha10):
    fit = 0
    for i in range(11):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_12(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7, amp8, tau8, pha8, amp9, tau9, pha9, amp10, tau10, pha10, amp11, tau11, pha11):
    fit = 0
    for i in range(12):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_13(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7, amp8, tau8, pha8, amp9, tau9, pha9, amp10, tau10, pha10, amp11, tau11, pha11, amp12, tau12, pha12):
    fit = 0
    for i in range(13):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_14(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7, amp8, tau8, pha8, amp9, tau9, pha9, amp10, tau10, pha10, amp11, tau11, pha11, amp12, tau12, pha12, amp13, tau13, pha13):
    fit = 0
    for i in range(14):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit


def cos_model_15(time, amp0, tau0, pha0, amp1, tau1, pha1, amp2, tau2, pha2, amp3, tau3, pha3, amp4, tau4, pha4, amp5, tau5, pha5, amp6, tau6, pha6, amp7, tau7, pha7, amp8, tau8, pha8, amp9, tau9, pha9, amp10, tau10, pha10, amp11, tau11, pha11, amp12, tau12, pha12, amp13, tau13, pha13, amp14, tau14, pha14):
    fit = 0
    for i in range(15):
        fit = fit + cos_model(time, eval("amp" + str(i)), eval("tau" + str(i)), eval("pha" + str(i)))
    return fit