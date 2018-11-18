import numpy as np
import matplotlib.pyplot as plt

# 極方程式
# 例:アルキメデスの渦巻線
if __name__ == '__main__':
    # os.chdir(os.path.join("/Users", "kenya", "keisan", "python", "00data"))
    plt.rcParams['axes.linewidth'] = 0  # x軸目盛りの太さ
    theta = np.arange(0.0, 2 * np.pi, 0.5)  # θの範囲を 0-8π ラジアン(4周分)とする
    plt.figure(figsize=(6, 6), dpi=100)  # A4余裕あり．かつ半分
    r = np.ones(len(theta))  # 極方程式を指定する。

    plt.polar(theta, r, color='w', marker='.', markersize=10, mec='r', mfc='r')  # 極座標グラフのプロット
    plt.ylim(0, 1.1)
    plt.yticks([0, 1])
    plt.tick_params(labelleft='off')
    plt.grid(which='major', color='black', linestyle='-')
    plt.xticks(np.array([0, np.pi / 2, np.pi, np.pi * 3 / 2, 0]))
    plt.show()
