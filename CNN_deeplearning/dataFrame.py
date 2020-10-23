import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob

#데이터 셋 만들기
data = glob('C:/Users/pjy/covid_analysis/total_test/0/*.jpg')
img_data = []
for img in data:
    path = Image.open(img)
    img_data.append(np.array(path))

img_data = np.array(img_data)
# *부분 데이터 총 수를 입력
img_data = list(img_data.reshape((1341,16384)))

column = []
for col in range(16384):
    c = 'pix:{}'.format(col+1)
    column.append(c)

   
df2 = pd.DataFrame(img_data,
    columns = column)
#csv형식의 DataFrame file 저장 위치
df2.to_csv('C:/Users/pjy/covid_analysis/total_test/*/*.csv')

