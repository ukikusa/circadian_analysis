# -*- coding: utf-8 -*-
"""Timeseries csv to phase, peak, r2 and."""


import os
import sys

import numpy as np
import pandas as pd

from analyser.FFT_nlls import data_norm, cos_fit
from analyser.peak_analysis import amp_analysis, phase_analysis
from analyser.make_figure import multi_plot


def circadian_analysis_for_csv(
    file, avg, save_f, dt_m=60, p_range=6, fit_range=4, pdf=False, offset=0, s=48, e=120
):
    """[summary]
    
    Args:
        file ([type]): [description]
        avg ([type]): [description]
        save_f ([type]): [description]
        dt_m (int, optional): [description]. Defaults to 60.
        p_range (int, optional): [description]. Defaults to 6.
        fit_range (int, optional): [description]. Defaults to 4.
        pdf (bool, optional): [description]. Defaults to False.
        offset (int, optional): [description]. Defaults to 0.
        s (int, optional): [description]. Defaults to 48.
        e (int, optional): [description]. Defaults to 120.
    
    Returns:
        [type]: [description]
    """
    _path, ext = os.path.splitext(file)
    if ext == ".csv":
        data_pd = pd.read_csv(file, dtype=np.float64)
    elif ext == ".tsv":
        data_pd = pd.read_csv(file, dtype=np.float64)
    else:
        sys.exit("Input is required csv or tsv file.")
    if len(data_pd.columns) == 1:
        sys.exit("Please use comma-delimited for csv and tab-delimited for tsv.")
    data_np = np.array(data_pd.values)

    # 解析
    peak_t, peak_v, data_phase, r2_np, peak_point, func, _ = phase_analysis(
        data_np, avg, dt_m, p_range, fit_range, offset
    )
    cv_np, _sd_np, _rms_np = amp_analysis(data_np, h_range=24 * 3)
    data_det, data_det_ampnorm = data_norm(data_np, dt=dt_m)
    fft_nlls = cos_fit(data_det, s=48, e=120, dt=dt_m)
    fft_nlls_ampnorm = cos_fit(data_det_ampnorm, s=48, e=120, dt=dt_m)

    time = np.arange(data_np.shape[0], dtype=np.float64) * dt_m / 60 + offset
    if pdf is not False:
        multi_plot(
            x=time,
            y=data_np,
            save_path=save_f + "peak_fit.pdf",
            peak=peak_point + int(avg * 0.5),
            func=func,
            r=p_range,
            label=data_pd.columns,
        )

    # dataframe に整形
    col = list(data_pd.columns)
    peak_t[peak_t == 0] = np.nan
    peak_t = pd.DataFrame(peak_t, columns=col)
    peak_v = pd.DataFrame(peak_v, columns=col)
    data_phase = pd.DataFrame(data_phase, columns=col, index=time)
    r2_pd = pd.DataFrame(r2_np, columns=col)
    cv_pd = pd.DataFrame(cv_np, columns=col, index=time)
    fft_nlls.index = col
    fft_nlls_ampnorm.index = col

    if save_f is not False:
        if not os.path.exists(os.path.dirname(save_f)):
            os.mkdir(os.path.dirname(save_f))
        if not os.path.exists(save_f):
            os.mkdir(save_f)
        peak_t.to_csv(save_f + "peak.csv")
        data_phase.to_csv(save_f + "phase.csv")
        peak_v.to_csv(save_f + "value.csv")
        r2_pd.to_csv(save_f + "r2.csv")
        cv_pd.to_csv(save_f + "cv.csv")
        np.savetxt(
            os.path.join(save_f, "data_det_ampnorm.csv"),
            data_det_ampnorm,
            delimiter=",",
        )
        fft_nlls.to_csv(os.path.join(save_f, str(s) + "h-" + str(e) + "h_fft_nlls.csv"))
        fft_nlls_ampnorm.to_csv(
            os.path.join(save_f, str(s) + "h-" + str(e) + "h_fft_nlls_ampnorm.csv")
        )
    return peak_t, peak_v, data_phase, r2_pd, cv_pd


if __name__ == "__main__":
    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python"))
    FIT_RANGE = 6
    P_RANGE = 6
    DT_M = 30
    DATA_F = os.path.join(
        "00data",
        "muranaka",
        "paper3_edit_data",
        "140315_Lg_p8L_AtCCA1ex4_NF1s_12DLL.csv",
    )
    SAVE = os.path.join("result", "muranaka", "140315_Lg_p8L_AtCCA1ex4_NF1s_12DLL", "")
    circadian_analysis_for_csv(
        DATA_F,
        avg=3,
        save_f=SAVE,
        p_range=P_RANGE,
        fit_range=FIT_RANGE,
        pdf=True,
        dt_m=DT_M,
    )
