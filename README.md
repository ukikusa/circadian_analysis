# circadian_analysis
リズム解析に関するプログラム開発です．  
FFT_nllsはanalysis/FFT_nlls.pyで行えますが，取り回しを良くする作業はまだしていません．  

ピクセルごとに位相と振幅(cv)を求める解析ができます．

# Dependency
python3 (3.7.0にて動作確認)  
library: `requirements.txt` 参照  

# set up
<details><summary>python3のインストールは，各自で．一応参考を畳んでおきます．</summary>
 mac    は下記をコマンドプロットにコピー&ペースト．

     /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
    brew install python3
    python3 -V

 windows10
    
  ~~https://qiita.com/taiponrock/items/f574dd2cddf8851fb02c~~  
  ~~とありますが上野は，VertualboxにUbuntuをインストールすることをおすすめします．~~  
  ~~https://qiita.com/ykawakami/items/4bae371932110b2e25e3~~  
  ~~設定済みのVertual box を渡すことも可能です．~~  
  Windows Subsystem for Linuxを使ってください．  
  http://www.aise.ics.saitama-u.ac.jp/~gotoh/UbuntuOnWSL.html
  Xサーバも入れる必要があります．  
  僕は[VcXsrv Windows X Server](https://sourceforge.net/projects/vcxsrv/)を入れました．
  https://qiita.com/optman/items/345df0d4d9188d4d0f90 
    
 </details>>
ライブラリのインストール (-pipは各自パスを入れてください．)

    pip3 install -r requirements.txt
備考: LABO MACに入っているのは pyenv-virtualenv  


# usage
(コマンドプロット，端末，シェルプロット，ターミナル等)での実行  
- 対話的に操作する場合  
python3  
と入力すると，python3に入れます．  
この際，バージョンが出ますが，python2.x.xでないことを確認してください．  
その上で，pythonのコードをコピペしていけば良いです．  

- ファイルごと実行する場合．  
```python3 "xxxx.py"```  
とすることで，"xxxx.py"内のコードを実行できます．


---
### 共通した使い方  
1. 必要なファイルを確認します．
    - ここでは "frond_phase.py" を例にします．
1. ログを残すため，必要なファイルを複製し，実行したソースコードを保存するようにしましょう．  
1. 複製したファイルの必要な部分を改変します．
    1. まず，ソースコードの入っているフォルダを指定します．  
    analyzerフォルダを右クリックし，詳細やプロパティを開くとpathが見えると思います．```sys.path.append(os.path.join('/hdd1, Users, kenya, Labo, keisan, python_sorce, analyser'))```
の```'/hdd1, Users, kenya, Labo, keisan, python_sorce'))```部分を各自の環境に合わせて変更してください．
    1. 作業するディレクトリを指定します．  
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

## ここに書くべきでないこと
一番使う材料  
MACと一緒．でもl．
