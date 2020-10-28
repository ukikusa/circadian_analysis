# -*-coding: utf-8 -*-
"""
タイムラプス撮影時に、ノイズ（宇宙線）除去のために、同じ条件下で２枚ずつ撮影しているとき、
その２枚のminimumを計算して、撮影条件ごとのフォルダーに保存するスクリプトです。
２波長測定系のように、１ループ内で大量の画像データがあるときに便利です。
撮影条件ごとのファイル名はresult0, result1,...となっています。

例えば、２波長測定系でsample 1 フィルターなし、緑フィルター、赤フィルター、と撮影した後、
sample 2で同様にフィルターなし、緑フィルター、赤フィルターの順に撮影した場合、analysis\min内に
result0: sample 1 Unfiltered image
result1: sample 1 Green-pass filtered image
result2: sample 1 Red-pass filtered image
result3: sample 2 Unfiltered image
result4: sample 2 Green-pass filtered image
result5: sample 2 Red-pass filtered image
のように保存されます。

<使い方>
(1) ターミナルでcalculate_min_img.pyが存在するディレクトリ内に移動する。
(2) ターミナルで、コマンドライン引数に気を付けながらcalculate_min_img.pyを実行する。
コマンドライン引数
-m: １ループで撮影される画像数。例えば、フィルターなし、緑フィルター、赤フィルターで同時に２サンプル撮影するときは１２枚。
--path, -p: 今回の実験のディレクトリのPATH。この下にanalysisファイルが生成され、minをとった画像が保存される。
--data, -d: file_dir下にある
--loop, -l: １ループの時間を何秒と想定しているか。デフォルトは3600秒。

実行コマンド例)
> python calculate_min_img.py -m 12 --path "C:\\Users\\watanabe\\Documents\\oyamalab\\data\\20201026"
"""

import os, sys
import glob
from PIL import Image, ImageMath
import numpy as np
import datetime
import argparse

#set the parser
parser = argparse.ArgumentParser(description="calculate the minimum value between 2 image and save in the proper file")
parser.add_argument('--path', '-p', help='parent directory name', type=str, default=os.getcwd())
parser.add_argument('-m', help='file per time', type=int, default=12)
parser.add_argument('--data', '-d', help='data directory name in the parent directory', type=str, default="data")
parser.add_argument('--loop', '-l', help='loop time in second', type=int, default=3600)

args = parser.parse_args()

M = args.m
file_dir = args.path
data = args.data
loop = args.loop


#load all the *.tif files in the data directory
os.chdir(os.path.join(file_dir, data))
listdir = glob.glob(os.path.join(r"*.tif"))
listdir.sort(key=lambda x: os.path.getmtime(x))
L = len(listdir)
N = L/M
print("File number in data directory:", L)
print("Time-lapse stack number: ", N)

# check if 1 loop of time-lapse is 60 min
os.chdir(os.path.join(file_dir, data))
loop_time = os.path.getmtime(listdir[M]) - os.path.getmtime(listdir[0])
if loop_time > loop*1.05 or loop_time < loop*0.95:
    print('Error: The time between 1 loop is {} second in your data.\nCheck "file per time" or "loop time in second"'.format(loop_time))
    sys.exit(1)
else:
    print("Time between 1 loop: {} sec".format(loop_time))

# analysisディレクトの中にminディレクトリをつくる
os.chdir(os.path.join(file_dir, data))
os.chdir("../")
os.getcwd()
if not os.path.exists('analysis'):
    os.mkdir("analysis")
    os.chdir("analysis")
    os.mkdir("min")
else:
    print("\nAnalysis folder already exist.")
    while True:
        ans = input("Are you sure you want to continue? [Y/n]").lower()
        if ans == 'n': sys.exit(1)
        if ans == 'y' or ans == '': break
        print("Please answer by \"y\" or \"n\"")

os.chdir(os.path.join(file_dir, "analysis", "min"))
dir_analysis = os.getcwd()

# minディレクトリ内にresultディレクトリをつくる
if not os.path.exists('result0'):
    for i in range(int(M/2)):
        os.mkdir("result" + str(i))

#二枚ずつ順番に写真をとってきてminimumを計算。
print("Calculating the minimum...")
for j in range(int(N)):
    for i in range(int(M/2)):
        # minをとる２枚の画像を取得してnumpy arrayに変換 -> minを計算
        os.chdir(os.path.join(file_dir, data))
        im1 = Image.open(listdir[M*j+(2*i)])
        im1 = np.array(im1)
        im2 = Image.open(listdir[M*j+(2*i)+1])
        im2 = np.array(im2)
        out = np.minimum(im1, im2)
        pil_img_f = Image.fromarray(np.uint16(out))
        
        # 対応するフォルダに保存
        os.chdir(os.path.join(dir_analysis, "result" + str(i)))
        pil_img_f.save("result{}_{}.tif".format(i, str(j).zfill(4)))
print("\nFinished! All images were stored in {}".format(dir_analysis))
