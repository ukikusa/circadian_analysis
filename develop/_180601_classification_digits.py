# coding: utf-8
import matplotlib.pyplot as plt
# sci-kit learn 読み込み
from sklearn import (svm, datasets, metrics)
import numpy as np
import image_analysis as im
import os
from sklearn.neural_network import MLPClassifier

# データセット画像ファイル読み込み


def make_dataset(learn_file, answer_file, question_file):
    digits = im.read_imgs(learn_file)
    # 表示
    # plt.matshow(digits.images[0], cmap="Greys")
    # plt.show()
    # 画像データを配列に変換 (numpy.ndarray型)
    X = digits
    # 画像データに対する数字ラベル (numpy.ndarray型)
    y = im.read_imgs(answer_file)
    X, y = X.reshape(len(X), -1), y.reshape(len(y), -1)
    # 訓練データを偶数行，テストデータを奇数行から採取
    X_train, y_train = X[0::2], y[0::2]
    X_test, y_test = X[1::2], y[1::2]
    # ## SVM (Support Vector Machine) で学習
    # clf = svm.SVC(gamma=0.001)
    clf = MLPClassifier(solver='sgd', random_state=0, max_iter=10000)
    # 訓練データとラベルで学習
    clf.fit(X_train, y_train)
    # 学習したモデルをテストデータに適用
    # question = im.read_imgs(question_file)
    accuracy = clf.score(X_test, y_test)
    print("再現率={accuracy}")
    # 学習済モデルを使ってテストデータを分類した結果を返す
    prediction = clf.predict(X_test)
    print("\nclassification report:")
    print(metrics.classification_report(y_test, prediction, digits=8))
    # precison: 適合率（精度）= 分類器で分類したデータの中で正解の割合
    # recall: 再現率 = 実際に分類されるデータの中で正しく分類されたものの割合
    # f1-score: 適合率と再現率の調和平均
    return prediction


if __name__ == '__main__':
    # os.chdir(os.path.join('/Users', 'kenya', 'keisan', 'python', '00data'))
    os.chdir(os.path.join('/hdd1', 'Users', 'kenya', 'Labo', 'keisan', 'python', '00data'))
    days = '170215-LL2LL-MVX'
    # ## ロジスティック回帰で学習
    anser = make_dataset(os.path.join(days, "raw_data", 'data_light', ''), os.path.join(days, 'edit_raw', 'div_frond', ''), os.path.join(days, 'raw_data', 'data_light', ''))
    print(anser.shape)
    anser2 = anser.reshape(anser.shape[0], int(np.sqrt(anser.shape[1])), -1).astype(np.uint)
    im.save_imgs('tmp', anser2)
sys.exit()
# In[8]:
