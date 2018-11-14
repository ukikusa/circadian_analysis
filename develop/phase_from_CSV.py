import numpy as np
import os
import image_analysis as im
import peak_analysis as pa
import pandas as pd
import sys


# dataはdata,avgで移動平均．前後p_range分より高い点を暫定ピークに．その時の移動平均がpeak_avg．fit_rangeでフィッティング

# pandasを用いてCSVファイルからデータを取り込み，ピーク，位相を出力するvar


def peak_find_fromCSV(file, avg, dT=60, p_range=6, fit_range=4, csv_save=False, pdf_save=False, offset=0, peak_v_range=0):
    # dataの取り込み,タブ区切りなら拡張子をtsv．　delimiter='\t'でできる気がする試してない．
    dataframe = pd.read_csv(file, dtype=np.float64, index_col=0)
    data = np.array(dataframe.values)
    # 時間の定義
    time = np.arange(data.shape[0], dtype=np.float64) * dT / 60 + offset
    peak_t, peak_v, data_phase, r2, peak_point, func, data_period = pa.phase_analysis(data, avg, dT, p_range, fit_range, offset, peak_v_range=peak_v_range)
    if pdf_save != False:
        pa.make_fig(x=time, y=data, save_path=pdf_save, peak=peak_point+int(avg*0.5), func=func, r=p_range, label=dataframe.columns)
    peak_t[peak_t == 0] = np.nan
    peak_t, peak_v, data_phase, r2 = pd.DataFrame(peak_t), pd.DataFrame(peak_v), pd.DataFrame(data_phase), pd.DataFrame(r2)
    col=list(dataframe.columns)
    peak_t.columns, peak_v.columns, data_phase.columns, r2.colums = col,col,col,col
    data_phase.index = time
    peak_t.to_csv(csv_save[:-4] + "peak.csv")
    data_phase.to_csv(csv_save[:-4] + "phase.csv")
    peak_v.to_csv(csv_save[:-4] + "value.csv")
    # pcov_csv.index = dataframe.index
    r2.to_csv(csv_save[:-4] + "r2.csv")
    return peak_t, peak_v, data_phase, r2

if __name__ == '__main__':
    os.chdir(os.path.join("/hdd1", "Users", "kenya", "Labo", "keisan", "python", "00data"))
    fit_range = 6
    p_range = 6
    dT = 60
    data_file = "2018-01-22-frond_photo_avg.csv"
    save_name = '2018-01-22-frond_photo_avg_peak'
    # peak = peak_find_fromCSV(data_file, avg=3, csv_save=save_name + ".csv", p_range=p_range, fit_range=fit_range, pdf_save=save_name+'.pdf', dT=dT)

    # sys.exit()

    # 以下，上野用
    ########################################
    days = "./170829-LL2LL-ito-MVX"
    dT = 60
    data_folder = os.path.join(days, "result")
    save_name = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    pdf_save=save_name+'.pdf'
    pdf_save=False
    print(save_name)
    peak = peak_find_fromCSV(os.path.join(data_folder, data_file), avg=3, csv_save=save_name+'csv', p_range=p_range, fit_range=fit_range, pdf_save=pdf_save, dT=dT, peak_v_range=0.01)
    #############################################
    sys.exit()
    days = "./170613-LD2LL-ito-MVX"
    dT = 60
    data_folder = os.path.join(days, "result")
    save_name = os.path.join(data_folder, 'pdf', os.path.splitext(data_file)[0])
    csv_save = save_name + ".csv"
    csv_save = False
    print(save_name)
    pdf_save = save_name+'pdf'
    peak = peak_find_fromCSV2(os.path.join(data_folder, data_file), avg=3, peak_avg=3, csv_save=csv_save, p_range=p_range, fit_range=fit_range, pdf_save=pdf_save, dT=dT, offset=1.5)

    #############################################
    days = "./170215-LL2LL-MVX"
    dT = 60 + 10 / 60
    data_folder = os.path.join(days, "result")
    save_name = os.path.join(data_folder, "pdf", os.path.splitext(data_file)[0])
    print(save_name)
    peak = peak_find_fromCSV2(os.path.join(data_folder, data_file), avg=3, peak_avg=3, csv_save=save_name + ".csv", p_range=p_range, fit_range=fit_range, pdf_save=save_name, dT=dT)
