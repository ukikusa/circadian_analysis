# -*-coding: utf-8 -*-
"""Today."""

import os
import sys

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

# ディレクトリを import の探索パスに追加
# sys.path.append('/hdd1/Users/kenya/Labo/keisan/python_sorce/develop')


def logistic(t, N0, K, r):
    """ロジスティック方程式

    Args:
        t: 計算する配列．
        N0: 初期値
        K: 密度効果
        r: 増殖率
    """
    t = np.array(t)
    N = K / ((1 + K / N0 - 1) * np.exp(-r * t) + 1)
    return N


def fitting(N, model=logistic, dt=60):
    """モデルに対するフィッティングを行う

    Args:
        N: 一次元 np.array
        model: logistic model (default: {logistic()})
        dt: [description] (default: {60})

    Returns:
        [description]
        [type]s
    """
    t = np.arange(N.size) * dt / 60
    fit = optimize.curve_fit(model, t, N, p0=[N[0], np.max(N), 0.5])
    func = fit[0]
    return func


def fittings(Ns, model=logistic, dt=60):
    data_n = Ns.shape[1]
    funcs = np.empty((data_n, 3), np.float64)
    for i in range(data_n):
        N = Ns[:, i]
        N = N[~np.isnan(N)]
        if N.shape[0] > 4 * 24 * 60 / dt:
            N = N[:int(4 * 24 * 60 / dt)]
        funcs[i] = fitting(N, model=model, dt=dt)
    return funcs


def make_fig(save, N, func, dt=60, title=False, model=logistic):
    t = np.arange(N.size) * dt / 60
    fig = plt.figure(figsize=(6, 4), dpi=100)
    plt.plot(t, N, lw=0.5, color='b')
    plt.plot(t, logistic(t, *func), lw=0.5, color='r')
    plt.legend(["data", "fittng"])
    if title is not False:
        plt.title(title, fontsize=6)
    plt.savefig(save, format='pdf')
    plt.close()
    return 0


def make_figs(save, data, funcs, dt=60, model=logistic):
    Ns = data.values
    data_n = Ns.shape[1]
    title = data.columns
    pp = PdfPages(save)
    for i in range(data_n):
        N = Ns[:, i]
        N = N[~np.isnan(N)]
        make_fig(pp, N, funcs[i], dt=dt, title=title[i], model=logistic)
    pp.close()
    return 0


def logistic_est(file, save, dt=60):
    data = pd.read_csv(folder)
    if os.path.exists(save) is False:
        os.makedirs(save)
    Ns = data.values
    # fittng
    funcs = fittings(Ns, model=logistic, dt=dt)
    make_figs(os.path.join(save, 'fitting.pdf'), data, funcs, dt=dt, model=logistic)  # 作図 
    func_pd = pd.DataFrame(funcs, columns=['N0', 'K', 'r'], index=data.columns)
    func_pd.to_csv(os.path.join(save, 'func.csv'))

if __name__ == "__main__":
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))
    # カレントディレクトリの変更．

    ####################### 保存するフォルダと，データが有るフォルダの設定．
    save = os.path.join('_190109', 'size_peaktopeak_nakazawa')
    folder = os.path.join('_190109', 'size_peaktopeak.csv')
    ####################### dtは分刻み
    dt=20
    #############################################

    logistic_est(folder, save, dt=20)
