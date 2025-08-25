"""
画像比較して異なる箇所を別画像で表示
↓を参考に
https://note.nkmk.me/python-opencv-numpy-image-difference/
"""

import cv2, os
import numpy as np
# from opencv_japanese import imread, imwrite

dirname =  os.path.dirname(__file__)

# img_1 = imread(dirname + '\\1_1.png')
# img_2 = imread(dirname + '\\1_2.png')

def np_imwrite(filename, img):
    # 画像データを uint8 (0-255) に変換し、クリッピング
    img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
    # imencode と .tofile で日本語パス対応書き込み
    cv2.imencode('.png', img_uint8)[1].tofile(filename)

img_1 = cv2.imdecode(np.fromfile(dirname + '\\diff_3_1.png', dtype=np.uint8), cv2.IMREAD_COLOR)
img_2 = cv2.imdecode(np.fromfile(dirname + '\\diff_3_2.png', dtype=np.uint8), cv2.IMREAD_COLOR)


height = img_1.shape[0]
width = img_1.shape[1]

img_size = (int(width), int(height))

# 画像をリサイズする
image1 = cv2.resize(img_1, img_size)
image2 = cv2.resize(img_2, img_size)

# ２画像の差異を計算
im_diff = image1.astype(int) - image2.astype(int)

# 単純に差異をそのまま出力する
np_imwrite(dirname + '/output/01_diff.png', im_diff)

# 差異が無い箇所を中心（灰色：128）とし、そこからの差異を示す
np_imwrite(dirname + '/output/02_diff_center.png', im_diff + 128)

# 差異が無い箇所を中心（灰色：128）とし、差異を2で割った商にする（差異を-128～128にしておきたいため）
im_diff_center = np.floor_divide(im_diff, 2) + 128
np_imwrite(dirname + '/output/03_diff_center.png', im_diff_center)
