# -*- coding: utf-8 -*-
"""Timeseries csv to phase, peak, r2 and."""


import os
import sys

import numpy as np
import pandas as pd

from analyser.peak_analysis import amp_analysis, phase_analysis
from analyser.make_figure import multi_plot


def circadian_analysis_for_csv(
    file, avg, dt_m=60, p_range=6, fit_range=4, csv_save=False, pdf_save=False, offset=0
):
    """[summary]

    Args:
        file ([type]): [description]
        avg ([type]): [description]
        dt_m (int, optional): [description]. Defaults to 60.
        p_range (int, optional): [description]. Defaults to 6.
        fit_range (int, optional): [description]. Defaults to 4.
        csv_save (bool, optional): [description]. Defaults to False.
        pdf_save (bool, optional): [description]. Defaults to False.
        offset (int, optional): [description]. Defaults to 0.

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
    peak_t, peak_v, data_phase, r2, peak_point, func, _ = phase_analysis(
        data_np, avg, dt_m, p_range, fit_range, offset
    )
    cv_np, _sd_np = amp_analysis(data_np, h_range=24 * 3)
    if pdf_save is not False:
        time = np.arange(data_np.shape[0], dtype=np.float64) * dt_m / 60 + offset
        multi_plot(
            x=time,
            y=data_np,
            save_path=pdf_save,
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
    r2 = pd.DataFrame(r2, columns=col)
    cv_pd = pd.DataFrame(cv_np, columns=col, index=time)

    if csv_save is not False:
        if not os.path.exists(os.path.dirname(csv_save)):
            os.mkdir(os.path.dirname(csv_save))
        peak_t.to_csv(csv_save + "peak.csv")
        data_phase.to_csv(csv_save + "phase.csv")
        peak_v.to_csv(csv_save + "value.csv")
        r2.to_csv(csv_save + "r2.csv")
        cv_pd.to_csv(csv_save + "cv.csv")
    return peak_t, peak_v, data_phase, r2, cv_pd


if __name__ == "__main__":
    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python"))
    FIT_RANGE = 6
    P_RANGE = 6
    DT_M = 60
    data_f = os.path.join(
        "00data",
        "muranaka",
        "paper3_edit_data",
        "140315_Lg_p8L_AtCCA1ex4_NF1s_12DLL.csv",
    )
    save = os.path.join("result", "muranaka", "140315_Lg_p8L_AtCCA1ex4_NF1s_12DLL", "")
    peak = circadian_analysis_for_csv(
        data_f,
        avg=3,
        csv_save=save,
        p_range=P_RANGE,
        fit_range=FIT_RANGE,
        pdf_save=save + "data.pdf",
        dt_m=DT_M,
    )
