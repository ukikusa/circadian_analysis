# circadian_analysis
リズム解析に関するプログラム開発です．  
FFT_nllsはanalysis/FFT_nlls.pyで行えますが，取り回しを良くする作業はまだしていません．  

ピクセルごとに位相と振幅(cv)を求める解析ができます．

# Dependency
python3 (3.7.0にて動作確認)  
library: `requirements.txt` 参照  

# set up
## python3のinstall
<details><summary>各自でググった方が楽しい．一応参考を畳んでおきます．</summary>

### Linux
  ggrks

### mac   
  homebrewを入れてない場合は入れる．そしたら簡単に入ります．  
  下記をコマンドプロットにコピー&ペースト．  

    /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew install python3
    python3 -V
  ただし，brew update をすることでpython3のversionが変わるとパスが壊れます．pyenvを使うなり，その都度ググるなりしてください．

### windows10   
  ~~windowsOSに直接pythonの環境設定をする？そんなめんどくさいことする奴いるの？~~  
  Windows Subsystem for Linuxを使ってください(windows7ならおとなしくVertualboxを使ってください)．  
  [参考](http://www.aise.ics.saitama-u.ac.jp/~gotoh/UbuntuOnWSL.html)  
  Xサーバも入れる必要があります．  
  僕は[VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)を入れました．  
  [参考](https://qiita.com/optman/items/345df0d4d9188d4d0f90)

### windows7  
  ~~windowsOSに直接pythonの環境設定をする？そんなめんどくさいことする奴いるの？~~  
  [https://qiita.com/taiponrock/items/f574dd2cddf8851fb02c]  
  とありますが上野は，VertualboxにUbuntuをインストールすることをおすすめします．  
  [https://qiita.com/ykawakami/items/4bae371932110b2e25e3]  
  設定済みのVertual box を渡すことも可能です．  
 </details>

## ライブラリのインストール 
  以下をコピペ．`requirements.txt`は各位pathを入力．  

    pip3 install -r requirements.txt
備考: LABO MACに入っているのは pyenv-virtualenv  


# usage
(コマンドプロット，端末，シェルプロット等)での実行  
Directoryは`circadian_analyser`に移動して実行してください．

    cd 'path'  

実行したいプログラム```xxxx.py```を走らせる  

    python3 "xxxx.py"
とすることで，"xxxx.py"内のコードを実行できます．

- guiとついているファイルはそのまま実行できます(作成中)．

- フロンドの移動補正 transform_GUI.py

---
## 以下　GUIファイルになっていないものを使用する場合．

1. ログを残すため，必要なファイルを複製し，実行したソースコードを保存するようにしましょう．(自動保存にプログラムを改変中)  
    1. 作業するディレクトリを指定します．(guiファイルの場合は不要)  
        ```os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python'))```を先と同様に変えてください．  
        今後の作業はここで指定したフォルダの中で行われます．  
        不要ならコメントアウトしてください．  
    1. 必要なパラメータを変更してください．
        dtやavg, foloderなどを変更することになると思います．
1. 変更したファイル内容を実行します．


### その他
- pathがわからない人は適当にググってください．3秒ででて3分で理解できます．
- 細かいパラメタの説明は書いていませんが色々変えたいときは python3 の環境に入り，import やfrom から始まっている行を全てコピペした後，
```help(関数名)```
と打てば，関数の説明が出ます．

## どのファイルで何ができるか．簡単に．
