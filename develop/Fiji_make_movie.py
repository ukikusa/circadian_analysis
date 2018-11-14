import os
import glob
from ij import IJ

# フロンド毎の画像を取得して，動画フォルダにまとめて保存
# グラフ化
os.chdir('/hdd1/Users/kenya/Labo/keisan/python/00data')
number = 168
# os.chdir('/home/kenya/Dropbox/Labo/python/00data')
days = ['./170215-LL2LL-MVX', './170613-LD2LL-ito-MVX', './170829-LL2LL-ito-MVX']
for day in days:
    frond_folder = day + '/frond'
    # for_movies = ['frond_lum','small_moved_mask_frond', 'small_moved_mask_frond_lum', 'moved_frond_lum', 'mask_frond']
    # for_movies = ['small_phase_color_mesh1_avg3', 'small_period_color_mesh1_avg3']
    for_movies = ['moved_mask_frond', 'moved_cg_mask_frond', 'mask_frond']
    color = False
    fps = str(24)

    for for_movie in for_movies:
        for i in glob.glob(frond_folder + '/*'):
            # 解析データのフォルダ
            folder = 'open=' + os.path.join(os.path.abspath(i), for_movie, '000.tif') + ' number=' + str(number) + ' sort'
            # read image
            IJ.run("Image Sequence...", folder)
            # set contrast
            if color is False:
                IJ.run("Enhance Contrast", "saturated=0.35")
            save_folder = os.path.join(day, 'result', 'movie', for_movie, i.split('/')[-1] + '.avi')
            if os.path.exists(os.path.dirname(save_folder)) is False:
                os.makedirs(os.path.dirname(save_folder))
            save = 'compression=JPEG frame=' + fps + ' save=' + os.path.abspath(save_folder)
            IJ.run("AVI... ", save)
            IJ.run('Close')
