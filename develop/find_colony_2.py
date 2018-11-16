import numpy as np
import os
import cv2
import glob
import sys
import math
import datetime
import image_analysis as im
from edit_img import label_imgs


def remove_back(folder, out_folder='/remove_back', back='white'):
    # folder内のデータの最大値または最小値を取り
    if os.path.exists(folder + out_folder + '/data_light_remove_back') is False:
        os.makedirs(folder + out_folder + '/data_light_remove_back')
    # 保存先フォルダの作成
    if os.path.exists(folder + '/raw_data/data_light') is False:
        print(folder + '/raw_data/data_light/*.tifがありません')
        sys.exit()
    img = im.read_imgs(folder + '/raw_data/data_light/')

    if os.path.exists(folder + out_folder + '/tmp_remove.tif') is False:
        # 画像の最大値を抽出する．
        if back == 'white':
            out_img = img.max(axis=0)
        elif back == 'black':
            out_img = img.min(axis=0)
        else:
            print('背景がwhiteかblackかを指定して')
            sys.exis()
        cv2.imwrite(folder + out_folder + '/tmp_remove.tif', out_img)
        print(out_folder + '/tmp_remove.tif を保存しました')
        input(out_folder + '/background.tif を作成してください>>')

    if os.path.exists(folder + out_folder + '/background.tif') is False:
        print('backgroundのスペル違うのでは')
        sys.exit()
    # imgをbackgroundを取り除いた画像に置き換える．
    img_back = cv2.imread(folder + out_folder + '/background.tif', cv2.IMREAD_GRAYSCALE |cv2.IMREAD_ANYDEPTH)
    if back == 'white':
        img_remove_back = - img + img_back
        img_remove_back[img > img_back] = np.min(img_remove_back)
    elif back == 'black':
        img_remove_back = img - img_back
        img_remove_back[img < img_back] = np.min(img_remove_back)
    else:
        print('背景がwhiteかblackかを指定して')
        sys.exis()
    # やっと保存するよ！
    for i in range(img.shape[0]):
        cv2.imwrite(folder + out_folder + '/data_light_remove_back/' + str(i).zfill(3) + '.tif', img_remove_back[i, :, :])
    print('背景を引いた画像を保存しました')
    # 保存した画像を返す．img[何枚目か,x?,y?]
    return img, img_remove_back


def threshold(img, ave=5, sd_x=3, kernel=3):# , color=15, space=51):
    # 二値化の関数.
    # 画像をガウシアンで平滑化し，大津ので二値化し，ノイズや針消去のため，縮小拡大．
    # （img, ave=Gaussianの範囲, sd_x=Gaussianのx方向標準偏差, kernel=縮小拡大のピクセル数）
    # returnは8bit．threshold済み画像
    if np.max(img) > 255 or np.min(img) >= 10:
        img8 = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    else:
        img8 = img
    # 移動平均を取るピクルス数を指定
    average = (ave, ave)
    # Gaussaianで平滑化以降
    img_gauss = cv2.GaussianBlur(img8, average, sd_x)
    # img_bilateral = cv2.bilateralFilter(img8, ave, sigmaColor=torakkinngucolor, sigmaSpace=space)
    # sigmaColorはどれだけ色が遠いものまで平滑化するか，sigmaspaceはガウシアンと同じようなもの，だと思う．
    # 二値化, thresholdtypeは大津（他に良い選択肢あるんかな
    img_threshold = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 針を消すために縮小し，拡大する．
    ker_one = np.ones((kernel, kernel), np.uint8)
    result = cv2.morphologyEx(img_threshold[1], cv2.MORPH_OPEN, ker_one)
    return result


def threshold_imgs(imgs, ave=5, sd_x=1, kernel=5):
    # 二値化とマージしたものを出力
    imgs_threshold = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        imgs_threshold[i, :, :] = threshold(imgs[i, :, :], ave=ave, sd_x=sd_x, kernel=kernel)
    return imgs_threshold


def find_edge_img(img, Canny_min=5, Canny_max=30, Gaussian_sd=3, Closing = False, Closing_kernel=3):
    # Canny法で輪郭抽出． Closingは回数． GaussianとClosingはFalseで処理なし． Closing_kernelはClosingの広さ
    # 次の処理のために8bit画像を作成
    if np.max(img) > 255 or np.min(img) >= 10:
        img8 = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    # Gaussian平滑化
    if Gaussian_sd is not False:
        img8_gauss = cv2.GaussianBlur(img8, (Gaussian_sd + 2, Gaussian_sd + 2), Gaussian_sd)
    else:
        img8_gauss = img8
    # Canny法． MAXの値より大きい輪郭は無条件でトリ，底からminの大きさにつなげていく．
    # 輪郭が白く出る（255）
    edge = cv2.Canny(img8_gauss, Canny_min, Canny_max)
    # 線を拡大する．　僅かに切れているところをつなぐため.
    if Closing is not False:
        # 拡大する太さを指定
        kernel = np.ones((Closing_kernel, Closing_kernel), np.uint8)
        for i in range(Closing):
            edge = cv2.dilate(edge, kernel)
            edge = cv2.erode(edge, kernel)
    img_edge_merge = img8
    img_edge_merge[edge == 255] = 255
    return edge, img_edge_merge


def find_edge_imgs(imgs, Canny_min=5, Canny_max=50, Gaussian_sd=3, Closing = 5, Closing_kernel=5):
    # find_edge_img を実行．
    imgs_edge = np.empty_like(imgs)
    imgs_edge_merge = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        imgs_edge[i, :, :], imgs_edge_merge[i, :, :] = find_edge_img(imgs[i, :, :], Canny_min=Canny_min, Canny_max=Canny_max, Gaussian_sd=Gaussian_sd, Closing=Closing, Closing_kernel=Closing_kernel)
    return imgs_edge, imgs_edge_merge


def find_frond_edge_img(img, Canny_min=5, Canny_max=30, sigmaColor=20, Closing = False, Closing_kernel=3):
    # Canny法で輪郭抽出． Closingは回数． GaussianとClosingはFalseで処理なし． Closing_kernelはClosingの広さ
    # 次の処理のために8bit画像を作成
    # 返り値は2つ．ひとつ目はエッジだけの情報．ふたつ目はマージした情報．
    if np.max(img) > 255 or np.min(img) >= 10:
        img8 = ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)
    else:
        img8 = img.astype(np.uint8)
    # Gaussian平滑化
    if sigmaColor is not False:
        # bilateralFilter
        # sigmaColorはどれだけ色が遠いものまで平滑化するか，sigmaspaceはガウシアンと同じようなもの，だと思う．
        img8_bilateral = cv2.bilateralFilter(img8, 11, sigmaColor=sigmaColor, sigmaSpace=9)
        # img8_bilateral = cv2.bilateralFilter(img8_bilateral, 5, sigmaColor=sigmaColor, sigmaSpace=3)
        # img8_bilateral = cv2.bilateralFilter(img8_bilateral, 5, sigmaColor=sigmaColor, sigmaSpace=3)
    else:
        img8_bilateral = img8
    # Canny法． MAXの値より大きい輪郭は無条件でトリ，底からminの大きさにつなげていく．
    # 輪郭が白く出る（255）
    edge = cv2.Canny(img8_bilateral, Canny_min, Canny_max)
    # 線を拡大する．　僅かに切れているところをつなぐため.
    if Closing is not False:
        # 拡大する太さを指定
        kernel = np.ones((Closing_kernel, Closing_kernel), np.uint8)
        for i in range(Closing):
            edge = cv2.dilate(edge, kernel)
            edge = cv2.erode(edge, kernel)
    img_edge_merge = img8
    img_edge_merge[edge == 255] = 255
    return edge, img_edge_merge


def find_frond_edge_imgs(imgs, Canny_min=5, Canny_max=50, sigmaColor=20, Closing = 5, Closing_kernel=5):
    # find_edge_img を実行．
    imgs_edge = np.empty_like(imgs)
    imgs_edge_merge = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        imgs_edge[i, :, :], imgs_edge_merge[i, :, :] = find_frond_edge_img(imgs[i, :, :], Canny_min=Canny_min, Canny_max=Canny_max, sigmaColor=sigmaColor, Closing=Closing, Closing_kernel=Closing_kernel)
    return imgs_edge, imgs_edge_merge


def frond_label_img(img_edge, img, connectivity=4, inversion=False, min_frond_stats=20):
    # ラベリングする．白い部分がラベルされるよ！ラベルしたい部分が黒かったらinversionをTrueに．
    # edge_imgがラベルする画像， imgの中央値か平均値を代入するよ
    img_edge = img_edge.astype(np.uint8)
    if inversion is True:
        img_edge = 255 - img_edge
    # ラベリングして，それぞれに平均値か中央値入れる！更に白黒に．
    # つながっているところをラベリングしていく．connetorakkinnguctivity が4なら斜めも線．8なら斜めはすり抜ける．
    # label に何箇所スペースがあったか． img_labelにラベリングされた画像, [始点x,y，長さx,y, 面積], その他
    label, img_label, img_stats, other = cv2.connectedComponentsWithStats(img_edge, connectivity=connectivity)

    # 何ピクセル以下のスペースを無視するか．
    tmp = img_stats[:, 4] > min_frond_stats
    # カラーにする面積．これ以下の面積はグレーで表示する．
    color_tmp = img_stats[:, 4] > 20
    color_label_square = int(sum(color_tmp))
    hue_number = 0
    # color画像用の場所づくり
    label_color = np.zeros((img_edge.shape[0], img_edge.shape[1], 3)).astype(np.uint8)
    for i in range(1, label):
        if tmp[i] == 1 and np.median(img[img_label == i]) != 0:
            if color_tmp[i] == 1:
                hue = math.floor(180/color_label_square) * hue_number
                label_color[img_label == i, :] = [hue, 180, 180]
                hue_number += 1
                print(hue)
            else:
                label_color[img_label == i, :] = [160, 2, 40]
    label_color = cv2.cvtColor(label_color, cv2.COLOR_HSV2BGR)
    # img_threshold = cv2.threshold(result8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return label_color, img_label, img_stats


def frond_label_imgs(imgs_edge, imgs, connectivity=4, inversion=False):
    # ラベリングする．白い部分がラベルされるよ！ラベルしたい部分が黒かったらinversionをTrueに．
    # edge_imgがラベルする画像， imgの中央値か平均値を代入するよ
    imgs_label_color = np.zeros((imgs_edge.shape[0], imgs_edge.shape[1], imgs_edge.shape[2], 3))
    imgs_result = np.empty_like(imgs)
    for i in range(imgs.shape[0]):
        imgs_label_color[i, :, :] = frond_label_img(imgs_edge[i, :, :], imgs[i, :, :], connectivity=connectivity, inversion=inversion)[0]
    return imgs_label_color, imgs_result


def make_color_img(img, color_n):  # 0が背景．1〜color_nまでのラベリング画像を入力．
    # color画像用の場所づくり
    img_color = np.zeros((img.shape[0], img.shape[1], 3)).astype(np.uint8)
    for i in range(1, color_n):
        hue = math.floor(180 / color_n) * i
        img_color[img == i, :] = [hue, 180, 180]
    img_color = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)
    return img_color


def make_color_imgs(imgs, color_n):
    imgs_color = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], 3)).astype(np.uint8)
    for i in range(imgs.shape[0]):
        imgs_color[i, :, :] = make_color_img(imgs[i, :, :], color_n=color_n)
    return imgs_color


day = '/' + str(datetime.date.today())

first = False

if first is True:
    os.chdir('/hdd1/kenya/Labo/keisan/python')

    # 以下そのうち関数にちゃんとしようね．
    folder = './img_ngazou170215'
    imgs, imgs_remove_back = remove_back(folder)

    # im.save_imgs(folder + '/data_light_remove_back', imgGaussian_sd)

    # imgs = im.read_imgs(folder + '/raw_data/data_light')
    # imgs = 65535 - imgs time = '4'
    # imgs = im.read_imgs(folder_im.read_imgs)
    imgs_edge, imgs_edge_merge = find_edge_imgs(imgs, Canny_min=5, Canny_max=30, Gaussian_sd=3, Closing=1, Closing_kernel=5)
    result = label_imgs(imgs_edge, imgs_remove_back, connectivity=4, inversion=True)

    time = '0'
    save_edge = folder + '/170330/edge' + time
    save_label_threshold = folder + '/170330/label_threshold' + time
    save_label = folder + '/170330/label' + time
    save_merge = folder + '/170330/edge_merge' + time
    # result = find_edge_imgs(imgs)
    # print(cv2.connectedComponentsWithStats(imgs_edge[0]))
    im.save_imgs(save_edge, imgs_edge)
    im.save_imgs(save_merge, imgs_edge_merge)
    im.save_imgs(save_label, result[1])
    im.save_imgs(save_label_threshold, result[0])

second = False
time = 6
if second is True:
    os.chdir('/hdd1/kenya/Labo/keisan/python')

    # 以下そのうち関数にちゃんとしようね．
    folder = './img_170215'
    imgs, imgs_remove_back = remove_back(folder)
    save_edge = folder + '/170330/edge' + str(time)
    save_label_threshold = folder + '/170330/label_threshold' + str(time)
    save_label = folder + '/170330/label' + str(time)
    save_merge = folder + '/170330/edge_merge' + str(time)
    if os.path.exists(save_merge) is True:
        sys.exit(save_merge + 'フォルダが存在するため，処理を停止します．')
    # im.save_imgs(folder + '/data_light_remove_back', img)

    # imgs = im.read_imgs(folder + '/raw_data/data_light')
    # imgs = 65535 - imgs
    # imgs = im.read_imgs(folder_im.read_imgs)
    # imgs_edge, imgs_edge_merge = find_edge_imgs(imgs, Canny_min=5, Canny_max=30, Gaussian_sd=3, Closing=1, Closing_kernel=5)        cv2.imwrite(save_folder + '/' + str(i).zfill(3) + '.tif', img[i, :, :])
    imgs_edge_merge = im.read_imgs(folder + '/170330/edge_merg    for i in range(imgs.shape[0]):e' + str(time-1))
    imgs_edge = np.zeros_like(imgs_edge_merge)
    imgs_edge[imgs_edge_merge == 255] = 255
    result = label_imgs(imgs_edge, imgs_remove_back, connectivity=4, inversion=True)


    # result = find_edge_imgs(imgs)
    # print(cv2.connectedComponentsWithStats(imgs_edge[0]))
    im.save_imgs(save_edge, imgs_edge)
    im.save_imgs(save_merge, imgs_edge_merge)
    im.save_imgs(save_label, result[1])
    im.save_imgs(save_label_threshold, result[0])


third = False
if third is True:
    os.chdir('/hdd1/kenya/Labo/keisan/python')

    # 以下フォルダの設定，画像の読み込み．
    folder = './img_170215'
    imgs_duckweed = im.read_imgs(folder + '/170330/label_threshold' + str(time))
    imgs = im.read_imgs(folder + '/raw_data/data_light')
    save_imgs_duckweed = folder + day + '/duckweed_threshold'
    save_imgs_merge = folder + day + '/duckweed_merge'
    imgs_duckweed_return = np.zeros_like(imgs_duckweed)
    imgs_duckweed_merge = np.zeros_like(imgs)
    # 以下，隙間を埋める作業を全部の画像に
    for i in range(imgs_duckweed.shape[0]):
        # 処理用の画像
        img_duckweed = imgs_duckweed[i, :, :]
        # 隙間を埋める膨張，縮小処理のカーネルを指定．　これは前の処理いますでしたのと同じ値にしたほうがいい気がする．
        Closing_kernel = 5
        kernel = np.ones((Closing_kernel, Closing_kernel), np.uint8)
        Closing=3
        for j in range(Closing):
            img_duckweed = cv2.dilate(img_duckweed, kernel)
            img_duckweed = cv2.erode(img_duckweed, kernel)
        # 最後に，縮小，膨張処理しておこう．意味は？知らん！！
        kernel_one = np.ones((1,1), np.uint8)
        img_duckweed = cv2.erode(img_duckweed, kernel_one)
        img_duckweed = cv2.dilate(img_duckweed, kernel_one)
        # これで，二値化画像の代入
        imgs_duckweed_return[i, :, :] = img_duckweed
        print(i)
    # 全ての画像に対して，マージを行う．
    imgs_duckweed_merge[imgs_duckweed_return!=0] = imgs[imgs_duckweed_return!=0]
    im.save_imgs(save_imgs_duckweed,imgs_duckweed_return)
    im.save_imgs(save_imgs_merge, imgs_duckweed_merge)


forth = False
time = '0'
if forth is True:
    os.chdir('/hdd1/kenya/Labo/keisan/python')

    # 以下フォルダの設定，画像の読み込み．
    folder = './img_170215'
    imgs_duckweed = im.read_imgs(folder + day + '/duckweed_merge')
    imgs = im.read_imgs(folder + '/raw_data/data_light')
    im.save_imgs_edge = folder + day + '/frond_edge' + time
    im.save_imgs_label = folder + day + '/frond_label' + time
    # エッジ抽出．ここでパラメータいじって精度あげとくと良い．ただ，縁が太くなると消すの面倒なので，カーネルサイズ上げるのは避けるべし．
    imgs_edge, imgs_edge_merge = find_frond_edge_imgs(imgs_duckweed, Canny_min=5, Canny_max=40, sigmaColor=15, Closing=1, Closing_kernel=3)
    # ラベリング
    imgs_label_color, imgs_frond_label = frond_label_imgs(imgs_edge, imgs_duckweed, connectivity=4, inversion=True)
    # マージ
    if np.max(imgs) > 255 or np.min(imgs) >= 10:
        imgs8 = ((imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs)) * 255).astype(np.uint8)
    else:
        imgs8 = imgs.astype(np.uint8)
    imgs8[imgs_edge == 255] = 255
    im.save_imgs(im.save_imgs_edge, imgs8)
    im.save_imgs(im.save_imgs_label,  imgs_label_color)


five = True
# 何回目の処理か．
# 1からスタート
time = 6
day = '/2017-04-05'
if five is True:
    os.chdir('/hdd1/kenya/Labo/keisan/python')
    folder = './img_170215'
    imgs = im.read_imgs(folder + '/raw_data/data_light')
    # 保存先の設定
    im.save_imgs_edge = folder + day + '/frond_edge' + str(time)
    im.save_imgs_label = folder + day + '/frond_label' + str(time)
    if os.path.exists(im.save_imgs_edge) is True:
        sys.exit(im.save_imgs_edge + 'フォルダが存在するため，処理を停止します．')

    # 以下フォルダの設定，画像の読み込み．

    imgs_edge_merge = im.read_imgs(folder + day + '/frond_edge' + str(time-1))
    imgs_duckweed = im.read_imgs(folder + day + '/duckweed_merge')
    # エッジ情報だけを抽出．
    imgs_edge = np.zeros_like(imgs_edge_merge)
    imgs_edge[imgs_edge_merge == 255] = 255
    # 元画像の読み込み（ifの外に出そう）

    # エッジ情報だけを抽出．
    imgs_label_color, imgs_frond_label = frond_label_imgs(imgs_edge, imgs_duckweed, connectivity=4, inversion=True)
    # マージ
    if np.max(imgs) > 255 or np.min(imgs) >= 10:
        imgs8 = ((imgs - np.min(imgs)) / (np.max(imgs) - np.min(imgs)) * 255).astype(np.uint8)
    else:
        imgs8 = imgs.astype(np.uint8)
    imgs8[imgs_edge == 255] = 255
    im.save_imgs(im.save_imgs_edge, imgs8)
    im.save_imgs(im.save_imgs_label,  imgs_label_color)



# これ以下，店の追跡
six = True
if six is True:
    os.chdir('/hdd1/kenya/Labo/keisan/python')
    folder = './img_170215'
    # ラベリングが終わった画像を取り込む．エッジと背景が0・それ以外は0で無い値なら良い
    # 白黒でもカラーでもOK
    labeled_folder = day + '/frond_label' + str(time)
    imgs = im.read_imgs(folder + labeled_folder)
    # 0以外のところを黒に．
    imgs[imgs != 0] = 255
    # ここで8bitにしておこう．無駄かも・
    imgs.astype(np.uint8)
    # とりあえず穴を消す．これでラベル化の処理がおわり．
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :]
        label, img_label, img_stats, other = cv2.connectedComponentsWithStats(img, connectivity=4)
        # 面積順にラベル
        # なお，背景の要素が0に入っているので，それは省く．
        index = np.argsort(img_stats[1:, 4])[::-1]
        # sortする
        img_label_sort = np.zeros_like(img_label)
        # img_label_sort[range(1,label)]
        for j in range(1, label):
            img_label_sort[img_label==j] = index[j-1]+1
        # それぞれのlabelだけで整形する．
        # 拡大縮小とかで輪郭きれたら嫌だなぁ．
        # 最後のラベルでｘを代入する．
        x = 100
        for j in range(1, label):
            img_tmp = np.zeros_like(img_label)
            if np.sum(img_label_sort==j) != 0:
                img_tmp[img_label_sort==j] = 255
                # 細かい線を消すための拡大縮小．
                img_tmp = img_tmp.astype(np.uint8)
                kernel_one = np.ones((2, 2), np.uint8)
                img_tmp = cv2.dilate(img_tmp, kernel_one)
                img_tmp = cv2.erode(img_tmp, kernel_one)
                img_tmp2 = img_tmp
                img_tmp = 255 - img_tmp
                # 内側に穴がないかどうかを調べて，あったら塗りつぶす．
                l_t, i_l_t, i_s_t, o_t = cv2.connectedComponentsWithStats(img_tmp, connectivity=4)
                if label != 1:
                    img_tmp2[i_l_t >= 2] = 255
                # ここまでで，フロンド抽出完了．これをx番目のフロンドとする．
                img_label_sort[img_tmp2==255] = x
                x += 1
        imgs[i, :, :] = img_label_sort
    imgs[imgs < 100] = 0
    test_folder = folder + day + '/test'
    im.save_imgs(test_folder, imgs)

    # ここから，やっと追跡．
    imgs = im.read_imgs(test_folder)
    imgs[imgs != 0] = 255  # もう一回ラベル撮り直すので．時間ちょっとかかるけど，しなくていいように考えるのめんどくさい．
    imgs_result = np.zeros_like(imgs)  # アウトプット用の容器
    # 最初の画像を設定
    img2 = imgs[0, :, :]
    label2, img_label_sort, img_stats_sort, centroids_sort = cv2.connectedComponentsWithStats(img2, connectivity=4)
    centroids_sort = centroids_sort[1:]
    imgs_result[0, :, :] = img_label_sort
    print(img_label_sort)
    for i in range(1, imgs.shape[0]):
        img = img2
        img2 = imgs[i, :, :]
        label, img_label, img_stats, centroids = [], [], [], []  # 変数を初期化
        # 追跡元の画像は一つ前で追跡が終わった画像．
        label, img_label, img_stats, centroids = label2, img_label_sort, img_stats_sort, centroids_sort
        img_stats_sort, centroids_sort = [], []  # 初期化
        # 新たに追跡する画像
        label2, img_label2, img_stats2, centroids2 = cv2.connectedComponentsWithStats(img2, connectivity=4)
        centroids2 = centroids2[1:]
        # 重心間の距離を図るよ！！
        # 0の位置には背景の重心が入ってるから無視．
        min_index, min_distance = [], []  # おまじない
        centroids, centroids2 = np.array(centroids), np.array(centroids2)
        for j in range(centroids.shape[0]):
            distance = []  # j番目のフロンドに対する次の画像のk番目のフロンドの距離を格納する．
            for k in range(centroids2.shape[0]):
                # フロンド間の距離を計算する
                distance.append(np.linalg.norm(centroids[j]-centroids2[k]))
            # j番目の位置にj番目のフロンドに対して距離が一番短いフロンドのインデックスを格納，そして距離を収納．
            min_index.append(distance.index(min(distance)))
            min_distance.append(min(distance))
        # 面倒なのでnp配列に変える．　更に格納庫を作る．まあ，おまじない．
        min_index = np.array(min_index)
        min_distance = np.array(min_distance)
        img_label_sort = np.zeros_like(img_label2)

        # フロンドのラベルを，前の画像と合わせるよ！
        label_number = 0  # 一枚前の画像でついているラベルを参照するため．
        for j in range(min_index.size):
            # 同じインデックスを複数回参照していたら，距離が近い方を取る．ソウじゃない方は無視！フロンドが消えた時に起こる．
            label_number += 1
            while not((img_label == label_number).any()):
                label_number += 1  # 前の画像にlabel_numberなフロンドがなかったらlabel_number増やす．
                print(label_number)
            same_flond = np.where(min_index==min_index[j])[0]  # ひとつのフロンドを複数のフロンドで取り合っていないか．
            print(min_distance[same_flond])
            print(min_distance[j] == min(min_distance[same_flond]))
            if min_distance[j] == min(min_distance[same_flond]):  # 取り合ってたら，距離が短い場合だけ次の処理をする．
                # 前のフロンドのインデックスを次の画像に代入する．

                img_label_sort[img_label2 == min_index[j] + 1] = label_number
                centroids_sort.append([centroids2[min_index[j]]])
        imgs_result[i,:,:] = img_label_sort
    imgs_color_result = make_color_imgs(imgs_result, 10)
    test_folder = folder + day + '/test2'
    labeled_folder = folder + '/labeled_img'
    im.save_imgs(test_folder, imgs_color_result)
    im.save_imgs(labeled_folder, imgs_result)
