# coding: utf-8
"""PCA."""
import os
import sys

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

import numpy as np

import pandas as pd  # pip3 install xlrd xlwt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def pca_xlsx(data_file, str_use='.*', sheet_name=0, index_col=0, header=0, threshold=10, scale=True, loading=0.9, c=True):
    """Pca.

    Args:
        data_file: FileのPath．重複は不可．
        str_use: 特定の文字列を含む列のみ，使用．正規表現 (default: {'.*.'})
        sheet_name: 取り込みシートを指定(0始まり)． (default: {0})
        index_col: 因子名がある列を指定(0始まり)． (default: {0})
        header: サンプル名の行を指定(0始まり)． (default: {0})
        threshold: 発現量 (default: {10})
        scale: 基本はTrue (default: {True})
        loading: 保存する因子負荷量 (default: {0.9})
        c: {list} 色付けしたければ0-1のリストを． (default: {True})
    """
    #############################
    # データの取り込み
    #############################
    root, ext = os.path.splitext(data_file)  # 拡張子とそれ以外に分ける．
    save_path = root + "_" + str_use.replace(".", "").replace("*", "")
    if ext == '.csv':
        data = pd.read_csv(data_file, index_col=index_col, header=header)
    elif ext == '.xlsx':
        data = pd.read_excel(data_file, sheet_name=sheet_name, index_col=index_col, header=header)
    else:
        sys.exit('csvファイルかxlsxを使ってください')
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    print(save_path + 'に保存します．')
    data = data.loc[:, data.columns.str.match(str_use)]
    ###############################
    # PCA
    ###############################
    data_cut = data[np.max(data, axis=1) > threshold]  # 閾値以下のm-RNAを消す．
    dat = data_cut.T
    if scale:  # スケーリング(標準化)
        ss = StandardScaler(with_mean=True, with_std=True)
        dat = ss.fit_transform(dat)
    pca = PCA(whiten=False, svd_solver='full')  # PCAの設定
    # n_components 0-1 なら累積寄与率の上限． 'mle' なら最尤推定で求める．
    pca_r = pca.fit(dat)  # 主成分分析の実行
    # print(np.max(dat_cut.values))
    pca_point = pca.transform(dat)  # PCA結果をplotしたもの．
    n = pca_r.n_components_  # めっちゃ使うので．
    ###############################
    # PCA結果をプロット
    ###############################
    if c is True:
        c = np.linspace(0, 1, n)

    def mk_fig(pca_point, n=n, save_pdf=os.path.join(save_path, 'pca.pdf'), x=0, c=c, cmap='jet', fontsize=4):
        pp = PdfPages(save_pdf)
        pc_i = ['PC' + str(i + 1) for i in range(n)]
        fig = plt.figure(figsize=(9, 7), dpi=100)
        if x == 0:
            x = [0] * n
        else:
            x = range(n)
        for i in range(n - 1):
            ax = plt.axes([0.1, 0.1, 0.9, 0.8])
            pc_i.append('PC' + str(i + 2))
            sc = ax.scatter(pca_point[:, x[i]], pca_point[:, i + 1], c=c, cmap='jet')
            for l, x_, y_ in zip(data.columns, pca_point[:, x[i]], pca_point[:, i + 1]):
                ax.annotate(l, (x_, y_), fontsize=fontsize)
            ax.set_xlabel(pc_i[x[i]], fontsize=9)
            ax.set_ylabel(pc_i[i + 1], fontsize=9)
            plt.colorbar(sc, aspect=100, shrink=1)
            plt.savefig(pp, format='pdf')
            plt.clf(fig)
        pp.close()

    mk_fig(pca_point, n=n, save_pdf=os.path.join(save_path, 'pca.pdf'), x=0, c=c, cmap='jet', fontsize=4)
    mk_fig(pca_point, n=n, save_pdf=os.path.join(save_path, 'pca_big_font.pdf'), x=0, c=c, cmap='jet', fontsize=9)
    mk_fig(pca_point, n=n, save_pdf=os.path.join(save_path, 'pca_2.pdf'), x='list', c=c, cmap='jet', fontsize=9)

    ################################
    # 因子寄与率をプロット
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = plt.axes()
    pc_i = ['PC' + str(i + 1) for i in range(n)]
    ax.bar(range(n), pca_r.explained_variance_ratio_, tick_label=pc_i, align="center")
    plt.savefig(os.path.join(save_path, 'importance.pdf'))
    plt.close()
    ###############################
    # 負荷量
    fc = pca.components_ * np.c_[np.sqrt(pca.explained_variance_ * (n - 1) / n)]  # 計算
    writer = pd.ExcelWriter(os.path.join(save_path, "loading_factor.xlsx"))
    for i in range(n):
        fc_i = pd.DataFrame({"Transcript_ID": data_cut.index, "loading_factor": fc[i]})
        fc_p = fc_i[fc_i['loading_factor'] > loading]
        fc_p = fc_p.sort_values("loading_factor", ascending=False)
        fc_m = fc_i[fc_i['loading_factor'] < loading * -1]
        fc_m = fc_m.sort_values("loading_factor", ascending=True)
        fc_p.to_excel(writer, sheet_name='PC' + str(i + 1) + 'plus', index=False, header=True)  # 保存
        fc_m.to_excel(writer, sheet_name='PC' + str(i + 1) + 'minus', index=False, header=True)
    writer.save()
