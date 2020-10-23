import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from glob import glob
from image_trains import convert_image_size

#Image pixel data -> array form 
#이미지 경로지정
path = glob('C:/Users/pjy/Desktop/covid/576013_1042828_bundle_archive/COVID-19 Radiography Database/NORMAL/*.png')
covid_path = path+path1+path2
normal_path = path
print(len(path))
train_N_img =glob('C:/Users/pjy/covid_analysis/pre_train/0/*.jpeg')
train_P_img =glob('C:/Users/pjy/covid_analysis/pre_train/1/*.jpeg')
test_N_img =glob('C:/Users/pjy/covid_analysis/pre_test/0/*.jpeg')
test_P_img =glob('C:/Users/pjy/covid_analysis/pre_test/1/*.jpeg')

print(len(train_N_img))
print(len(train_P_img))
print(len(test_N_img))
print(len(test_P_img))

path_total = covid_path+normal_path
#이미지 픽셀값을 array형태로 저장하고 list에 담아주기
convert_image_size(path)
height = []
width = [] 
for img_name in path_total:
    img_pil = Image.open(img_name).convert("L")
    img_num = np.array(img_pil)
    # img_pil.append(img_num)
    h,w = img_num.shape
    height.append(h)
    width.append(w)

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.hist(width)
plt.title('width')

plt.subplot(122)
plt.hist(height)
plt.title('height')

plt.show()


                