import numpy as np
import os
import matplotlib.pyplot as plt
import image_analysis as im
import peak_analysis as pa
import pandas as pd
from make_phase_img import phase_fit

# dataはdata,avgで移動平均．前後p_range分より高い点を暫定ピークに．その時の移動平均がpeak_avg．fit_rangeでフィッティング

# pandasを用いてCSVファイルからデータを取り込み，ピーク，位相を出力するvar


def peak_find_fromCSV2(file, avg, dt=60, peak_avg=3, p_range=6, fit_range=4, csv_save=False, pdf_save=False, offset=0):
    # dataの取り込み,タブ区切りなら拡張子をtsv．　delimiter='\t'でできる気がする試してない．
    dataframe = pd.read_csv(file, dtype=np.float64, index_col=0)
    data = np.array(dataframe.values)
    # 時間の定義
    time = np.arange(data.shape[0], dtype=np.float64) * dt / 60 + offset
    time_move = time[int(avg / 2): data.shape[0] - int(avg / 2)]
    # peakや時間を出力する箱を作る．
    data_phase = np.empty_like(data, dtype=np.float64)
    peak_time = np.zeros((int(data.shape[0] / p_range), data.shape[1]))
    peak_value = np.zeros_like(peak_time)
    func_value = np.zeros((int(data.shape[0] / p_range), data.shape[1], 3))
    pcov = np.zeros((int(data.shape[0] / p_range) * 3, data.shape[1] * 3))
    # それぞれの時系列に対して処理を回す．ここでピーク抽出やグラフ作成を終わらす．
    if pdf_save is not False:
        plt.rcParams['font.size'] = 11
        if os.path.exists(pdf_save) is False:
            os.makedirs(pdf_save)
        # plt.figure(figsize=(8.27, 11.69), dpi=100)
        plt.figure(figsize=(6.5, 9), dpi=100)
    for i in range(data.shape[1]):
        if pdf_save is not False:
            # pdf一ページに対して，配置するグラフの数．配置するグラフの場所を指定．
            plt.subplot(7, 4, np.mod(i, 28) + 1)
        # peakを推定する．func_valueで計算．
        peak_time_tmp, peak_value_tmp, func_value_tmp, peak_point, pcov_tmp = pa.peak_find(
            data[:, i], time=time_move, avg=avg, peak_avg=peak_avg, p_range=p_range, fit_range=fit_range, pdf_save=pdf_save, label=dataframe.columns[i])
        if pdf_save is not False:
            # x軸の調整
            plt.ylim([0, 10])
            plt.tick_params(labelbottom='off', labelleft='off')
            plt.xlim([0, 192])
            plt.xticks(np.arange(0, 192, 24))
            plt.vlines(np.arange(0, 192, 24), 0, 10, colors="k", linestyle="dotted", label="", lw=0.5)
            plt.legend(loc='best', frameon=1, framealpha=1, handlelength=False, markerscale=False, fontsize=7, edgecolor='w')
            if np.mod(i, 28) == 27:
                # レイアウト崩れを自動で直してもらう
                plt.tight_layout()
                # 保存
                plt.savefig(os.path.join(pdf_save, str(i) + ".pdf"))
                plt.close()
                plt.figure(figsize=(6.5, 9), dpi=100)
            # peak_time.append(peak_time_tmp)
            # peak_value.append(peak_value_tmp)
            # print(peak_time_tmp)
        peak_time[0:len(peak_time_tmp), i] = peak_time_tmp
        if len(peak_time_tmp):
            pcov[0:len(peak_time_tmp) * 3, i * 3:i * 3 + 3] = np.reshape(pcov_tmp, (-1, 3))
        data_phase[:, i] = phase_fit(peak_time_tmp, data.shape[0], dt=dt)
    if np.mod(i, 28) != 27:
        plt.rcParams['font.size'] = 11
        plt.tight_layout()
        plt.savefig(os.path.join(pdf_save, str(i) + ".pdf"))
        plt.close()
    peak_time[peak_time == 0] = np.nan
    peak_time, data_phase = pd.DataFrame(peak_time), pd.DataFrame(data_phase)
    peak_time.columns, data_phase.columns = dataframe.columns, dataframe.columns
    data_phase.index = time
    peak_time.to_csv(csv_save)
    data_phase.to_csv(csv_save[:-4] + "phase.csv")
    pcov = pd.DataFrame(pcov)
    # pcov_csv.index = dataframe.index
    pcov.to_csv(csv_save[:-4] + "pcov.csv")
    return peak_value


if __name__ == '__main__':
    if 1 == 1:
        # os.chdir(os.path.join("/Users", "kenya", "keisan", "python", "00data"))
        os.chdir(os.path.join("/hdd1", "kenya", "Labo", "keisan", "python", "00data"))
        fit_range = 6
        p_range = 6

        data_file = "2018-01-22-frond_photo_avg.csv"

        ########################################
        days = "./170829-LL2LL-ito-MVX"
        dt = 60
        data_folder = os.path.join(days, "result")
        save_name = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
        print(save_name)
        peak = peak_find_fromCSV2(os.path.join(data_folder, data_file), avg=3, peak_avg=3, csv_save=save_name + ".csv", p_range=p_range, fit_range=fit_range, pdf_save=save_name, dt=dt)

        #############################################
        days = "./170613-LD2LL-ito-MVX"
        dt = 60
        data_folder = os.path.join(days, "result")
        save_name = os.path.join(data_folder, 'pdf', os.path.splitext(data_file)[0])
        print(save_name)
        peak = peak_find_fromCSV2(os.path.join(data_folder, data_file), avg=3, peak_avg=3, csv_save=save_name + ".csv", p_range=p_range, fit_range=fit_range, pdf_save=save_name, dt=dt, offset=1.5)

        #############################################
        days = "./170215-LL2LL-MVX"
        dt = 60 + 10 / 60
        data_folder = os.path.join(days, "result")
        save_name = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
        print(save_name)
        peak = peak_find_fromCSV2(os.path.join(data_folder, data_file), avg=3, peak_avg=3, csv_save=save_name + ".csv", p_range=p_range, fit_range=fit_range, pdf_save=save_name, dt=dt)
