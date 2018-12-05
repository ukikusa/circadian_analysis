import numpy as np
import os
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
import pandas as pd
import matplotlib.cm as cm
import sys


# pandasを用いてCSVファイルからデータを取り込み，ピークを出力するvar
def csv2fig(file, pdf_save, avg, dt, ymax, offset=0, loc="upper left", use_data='all'):
    # dataの取り込み,タブ区切りなら拡張子をtsv．　delimiter='\t'でできる気がする試してない．
    dataframe = pd.read_csv(file, dtype=float, index_col=0)
    if use_data is not 'all':
        dataframe = dataframe[use_data]
    data = np.array(dataframe.values)
    if ymax > 10000:
        data = data / np.power(10, int(np.log10(ymax)) - 4)
        ymax = ymax / np.power(10, int(np.log10(ymax)) - 4)
    # 時間の定義
    time = np.arange(data.shape[0], dtype=float) * dt / 60 + offset
    if avg is not False:
        time_move = time[int(avg / 2): data.shape[0] - int(avg / 2)]
    else:
        time_move = time
        # peakや時間を出力する箱を作る．
    plt.rcParams['font.size'] = 11
    if os.path.exists(pdf_save) is False:
        os.makedirs(pdf_save)
    # plt.figure(figsize=(8.27, 11.69), dpi=100) # 余白なしA4
    plt.figure(figsize=(6.5, 9), dpi=100)  # A4 余白を持って
    for i in range(data.shape[1]):
        # pdf一ページに対して，配置するグラフの数．配置するグラフの場所を指定．
        plt.subplot(7, 4, np.mod(i, 28) + 1)
        # peakを推定する．func_valueで計算．
        if avg is not False:
            data_move = pa.moving_avg(data[:, i], avg=avg)
        else:
            data_move = data[:, i]
        # x軸の調整
        plt.plot(time_move, data_move, lw=1, label=dataframe.columns[i])
        plt.ylim([0, ymax])
        # plt.ylim(bottom=0)
        plt.tick_params(labelbottom='off', labelleft='off')
        plt.tick_params(right="off", top="off")
        # plt.xlabel('time'
        plt.xlim([0, 192])
        plt.xticks(np.arange(0, 192, 24 * dt / 60))
        plt.legend(loc='best', frameon=1, framealpha=1, handlelength=False, markerscale=False, fontsize=7, edgecolor='w')
        plt.vlines(np.arange(0, 192, 24 * dt / 60), 0, ymax, colors="k", linestyle="dotted", label="", lw=0.5)

        if np.mod(i, 28) == 27:
            # レイアウト崩れを自動で直してもらう
            plt.tight_layout()
            # 保存
            plt.savefig(os.path.join(pdf_save, str(i) + ".pdf"))
            plt.close()
            plt.figure(figsize=(6.5, 9), dpi=100)

    if np.mod(i, 28) != 27:
        plt.rcParams['font.size'] = 11
        plt.tight_layout()
        plt.savefig(os.path.join(pdf_save, str(i) + ".pdf"))
        plt.close()
    return 0


def csv2fig_all(file, pdf_save, avg, dt, ymax, offset=0, loc="upper left", use_data=False, color=False):
    # dataの取り込み,タブ区切りなら拡張子をtsv．　delimiter='\t'でできる気がする試してない．
    dataframe = pd.read_csv(file, dtype=float, index_col=0)

    if use_data is not False:
        dataframe = dataframe.ix[:, use_data]
    data = np.array(dataframe.values)
    if ymax > 10000:
        data = data / np.power(10, int(np.log10(ymax)) - 4)
        ymax = ymax / np.power(10, int(np.log10(ymax)) - 4)
    # 時間の定義
    time = np.arange(data.shape[0], dtype=float) * dt / 60 + offset
    if avg is not False:
        time_move = time[int(avg / 2): data.shape[0] - int(avg / 2)]
    else:
        time_move = time
    plt.rcParams['font.size'] = 11
    if os.path.exists(pdf_save) is False:
        os.makedirs(pdf_save)
    # plt.figure(figsize=(8.27, 11.69), dpi=100)#余裕なしA4
    plt.figure(figsize=(6, 4), dpi=100)  # A4余裕あり．かつ半分
    plt.axes(axisbg="0.7")
    for i in range(data.shape[1]):
        # pdf一ヘーシに対して，配置するグラフの数．配置するクラフの場所を指定．
        if avg is not False:
            data_move = pa.moving_avg(data[:, i], avg=avg)
        else:
            data_move = data[:, i]
        # x軸の調整
        if use_data is False:
            plt.plot(time_move, data_move, linewidth=1, color=cm.brg(1 - i / data.shape[1]), label=dataframe.columns[i])
        else:
            plt.plot(time_move, data_move, linewidth=1, color=color[i], label=dataframe.columns[i])
    plt.ylim([0, ymax])
    # plt.ylim(bottom=0)
    plt.tick_params(labelbottom='off', labelleft='off')
    plt.tick_params(right="off", top="off")
    plt.xlim([0, 192])
    plt.xticks(np.arange(0, 192, 24))
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=1, framealpha=1, markerscale=False, fontsize=7, edgecolor='w')
    # plt.legend(loc='upper right', handlelength=False,  markerscale=3)  # , bbox_to_anchor=(1.05, 0.5, 0.5, .100), borderaxespad=0.,) #frameon=0, handlelength=False, markerscale=False)
    plt.vlines(np.arange(0, time[-1], 24), 0, ymax, colors="k", linestyle="dotted", label="", lw=1)
    if loc == "lower right":
        plt.text(time[-1] - int(time[-1] / 2), int(ymax / 10), dataframe.columns[i])
    elif loc == "upper left":
        plt.text(int(time[-1] / 15), ymax - int(ymax / 6), dataframe.columns[i])
    else:
        pass
        # レイアウト崩れを自動で直してもらう
    # plt.tight_layout()
    # 保存
    plt.savefig(os.path.join(pdf_save, str(i) + ".pdf"), bbox_inches='tight')
    plt.close()
    return 0


if __name__ == '__main__':
    # os.chdir(os.path.join("/Users", "kenya", "keisan", "python", "00data"))
    os.chdir(os.path.join("/hdd1", "kenya", "Labo", "keisan", "python", "00data"))
    ############################################
    ################ 総発光量 ##################
    ############################################
    data_file = "2018-01-22-frond_photo_sum.csv"
    avg = 3
    ymax = 40000
    loc = "out right"
    loc = "def"

    days = "./170829-LL2LL-ito-MVX"
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)

    ############################################
    ################# 面積 #####################
    ############################################
    data_file = "2018-01-22-frond_area.csv"
    avg = 3
    ymax = 6000
    loc = "out right"

    days = "./170829-LL2LL-ito-MVX"
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)

    ############################################
    ################# 平均 #####################
    ############################################
    data_file = "2018-01-22-frond_photo_avg.csv"
    avg = 3
    ymax = 10
    loc = "out right"

    days = "./170829-LL2LL-ito-MVX"
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "all_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)

    ############################################
    ################# 面積 #####################
    ############################################

    data_file = "2018-01-22-frond_area.csv"
    avg = 3
    ymax = 6000

    days = "./170829-LL2LL-ito-MVX"
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset)

    ################################################
    ################# 総発光量 #####################
    ################################################

    data_file = "2018-01-22-frond_photo_sum.csv"
    avg = 3
    ymax = 40000
    loc = "lower right"
    loc = "def"

    days = "./170829-LL2LL-ito-MVX"
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=offset, loc=loc)

    ############################################
    ################ 総発光量 ##################
    ############################################
    # color = np.array([[0, 255, 255], [255, 255, 0], [0, 128, 0], [0, 0, 255]]) / 255
    color = np.array([[255, 255, 0], [0, 255, 255], [0, 128, 0], [255, 0, 0]]) / 255
    data_file = "2018-01-22-frond_photo_sum.csv"
    avg = 3
    ymax = 40000
    loc = "out right"
    loc = "def"

    days = "./170829-LL2LL-ito-MVX"
    use_data = np.array(['01', '02', '03', '04'])
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    use_data = np.array(['01', '02', '03', '04'])
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    use_data = np.array(['01', '02', '03', '05'])
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)

    ############################################
    ################# 面積 #####################
    ############################################
    data_file = "2018-01-22-frond_area.csv"
    avg = 3
    ymax = 6000
    loc = "out right"
    use_data = np.array(['01', '02', '03', '04'])
    days = "./170829-LL2LL-ito-MVX"
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    use_data = np.array(['01', '02', '03', '04'])
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    use_data = np.array(['01', '02', '03', '05'])
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)

    ############################################
    ################# 平均 #####################
    ############################################
    data_file = "2018-01-22-frond_photo_avg.csv"
    avg = 3
    ymax = 10
    loc = "out right"
    use_data = np.array(['01', '02', '03', '04'])
    days = "./170829-LL2LL-ito-MVX"
    dt = 60
    offset = 0
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)
    #############################################
    days = "./170613-LD2LL-ito-MVX"
    dt = 60
    offset = 1.5
    use_data = np.array(['01', '02', '03', '04'])
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)
    #############################################
    days = "./170215-LL2LL-MVX"
    dt = 60 + 10 / 60
    offset = 0
    use_data = np.array(['01', '02', '03', '05'])
    data_folder = os.path.join(days, "result")
    pdf_save = os.path.join(data_folder, "pdf", "1-4_" + os.path.splitext(data_file)[0])
    print(pdf_save)
    csv2fig_all(os.path.join(data_folder, data_file), pdf_save, avg=avg, dt=dt, ymax=ymax, offset=0, loc=loc, use_data=use_data, color=color)
