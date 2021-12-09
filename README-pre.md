# circadian_analysis

リズム解析に関するプログラム開発です．  
FFT_nllsはanalysis/FFT_nlls.pyで行えますが，取り回しを良くする作業はまだしていません．  

ピクセルごとに位相と振幅(cv)を求める解析ができます．

## Dependency

python3 (3.7.0にて動作確認)  
library: `Pipfile` 参照  
pipenv にて環境再構築可能．  
~~記事を書くのめんどくさいので~~pipenvで検索して  
僕はpipenv + pyenv で環境を作っています．

## set up

### python3のinstall

<details><summary>参考を畳んでおきます．</summary>

#### Linux

ほとんどの場合，デフォルトで入っています．

#### mac

  homebrewを入れてない場合は入れる．そしたら簡単に入ります．  
  下記をコマンドプロットにコピー&ペースト．  

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew install python3
    python3 -V
  ただし，brew update をすることでpython3のversionが変わるとパスが壊れます．  
  pyenvの使用がおすすめです．

#### windows10

  windowsOSに直接pythonの環境設定をするとバグの元！  
  Windows Subsystem for Linuxがお薦めです．  
  [導入の際，参考にした記事](http://www.aise.ics.saitama-u.ac.jp/~gotoh/UbuntuOnWSL.html)  
  Xサーバも入れる必要があります．  
  僕は[VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)を入れました．  
  [参考にした記事](https://qiita.com/optman/items/345df0d4d9188d4d0f90)

#### windows7  

  VertualboxにUbuntuをインストールすることをおすすめします．  
  [参考にした記事](https://qiita.com/ykawakami/items/4bae371932110b2e25e3)  
  設定済みのVertual box を渡すことも可能です．  
</details>

### ライブラリのインストール

以下をコピペ．`path`はソースコードを入れた場所．  

    cd path  # コードのある場所に移動
    pip install pipenv  # 環境管理ソフト
    pipenv install  # ライブラリのインストール
備考: LABO MACに入っているのは pyenv-virtualenv  

## usage

(コマンドプロット，端末，シェルプロット等)での実行  
Directoryは`circadian_analyser`に移動して実行してください．

    cd 'path'  

実行したいプログラム```xxxx.py```を走らせる  

    python3 "xxxx.py"
とすることで，"xxxx.py"内のコードを実行できます．

- guiとついているファイルはそのまま実行できます(作成中)．
- フロンドの移動補正 transform_GUI.py

---

### 以下　GUIファイルになっていないものを使用する場合

1. ログを残すため，必要なファイルを複製し，実行したソースコードを保存するようにしましょう．(自動保存にプログラムを改変中)  
    1. 作業するディレクトリを指定します．(guiファイルの場合は不要)  
        ```os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))```を先と同様に変えてください．  
        今後の作業はここで指定したフォルダの中で行われます．  
        不要ならコメントアウトしてください．  
    1. 必要なパラメータを変更してください．
        dtやavg, foloderなどを変更することになると思います．
1. 変更したファイル内容を実行します．


#### その他

- pathがわからない人は適当にググってください．3秒ででて3分で理解できます．
- 細かいパラメタの説明は書いていませんが色々変えたいときは python3 の環境に入り，import やfrom から始まっている行を全てコピペした後，
```help(関数名)```
と打てば，関数の説明が出ます．

### どのファイルで何ができるか．簡単に．

#### phase_from_CSV

ST等のcsv, tsv形式で保存された時系列データを解析する．

- 出力
  - peak.csv : 二次関数fittingにより推定されたpeak list
  - phase.npy: peak listを使用し，位相を生成したもの
  - fft : fft_nlls法で求めた周期と位相

Analyze timesiries of csv, tsv file.

- Returns
  - peak.csv : a list of peak time estimated by a local quadratic curve fitting
  - phase.npy: phase of each time defined by the linear interpolation of peaks
  - fft : Period and phase estimated by Fast Fourier transform–nonlinear least squares (FFT-NLLS)
