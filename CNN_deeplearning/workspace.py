 #라이브러리 호출!!
import os, sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import cv2
from glob import glob
from PIL  import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import itertools
print(cv2.__version__)

# print(os.listdir('C:/Users/pjy/covid_analysis/val/'))

## 데이터 준비
train_norm = 'C:/Users/pjy/covid_analysis/train/train_norm.csv' 
train_PNE= 'C:/Users/pjy/covid_analysis/train/PNE.csv' 
test_norm = 'C:/Users/pjy/covid_analysis/test/test_norm.csv' 
test_PNE= 'C:/Users/pjy/covid_analysis/test/test_PNE.csv' 
val_norm = 'C:/Users/pjy/covid_analysis/val/val_norm.csv' 
val_PNE ='C:/Users/pjy/covid_analysis/val/val_PNE.csv' 

#DataFrame 형태 데이터 타입을 array로 변환하는 함수, 학습 시 array 타입으로 넣어야한다.
def array_data(shuffle_data):
    change_np = shuffle_data.to_numpy()
    array_val = change_np[:,1:]
    return array_val

#Labeling
def csv_load(file,label_val):
    train_norm = pd.read_csv(file)
    train_norm['label'] = '{}'.format(label_val)
    print(train_norm)
    return train_norm


train_N, test_N,val_N = csv_load(train_norm,0),csv_load(test_norm,0),csv_load(val_norm,0)
train_P,test_P,val_P = csv_load(train_PNE,1),csv_load(test_PNE,1),csv_load(val_PNE, 1)

#훈련데이터 만들기
train_X = pd.concat([train_N,train_P])
test_X = pd.concat([test_N,test_P])
#데이터 섞기
train_X_shuffle = train_X.sample(frac=1)
test_X_shuffle = test_X.sample(frac=1)

train_np = array_data(train_X_shuffle)
test_np = array_data(test_X_shuffle)

train_x ,train_y = train_np[:,:-1],train_np[:,-1]
test_x ,test_y = test_np[:,:-1],test_np[:,-1]

train_y,test_y = np.array(train_y, dtype=int), np.array(test_y,dtype=int)

#정규화
Norm_train_x , Norm_test_x = train_x/255.0,test_x/255.0
Norm_train_x,Norm_test_x =np.array(Norm_train_x,dtype=float),np.array(Norm_test_x,dtype=float)

#모델에 적용 시키기 위하여 reshape(batch_size,height,width,channel)
Norm_train_x= Norm_train_x.reshape(5331,128,128,1)
Norm_test_x= Norm_test_x.reshape(639,128,128,1)

train_Y = tf.keras.utils.to_categorical(train_y,num_classes=2)
test_Y = tf.keras.utils.to_categorical(test_y,num_classes=2)

# 모델구축
model = tf.keras.Sequential([
    #feature
    tf.keras.layers.Conv2D(input_shape=(128,128,1),kernel_size=(3,3),filters=32,padding='SAME',activation='relu'), 
    tf.keras.layers.Conv2D(kernel_size=(3,3),filters=32,padding='SAME',activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(rate=0.25),
    
    tf.keras.layers.Conv2D(kernel_size=(3,3),filters=64,padding='SAME',activation='relu'),
    tf.keras.layers.Conv2D(kernel_size=(3,3),filters=64,padding='SAME',activation='relu'),
    tf.keras.layers.MaxPool2D(strides=(2,2)),
    tf.keras.layers.Dropout(rate=0.25),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=512,activation='relu'),
    tf.keras.layers.Dropout(rate=0.25),
    tf.keras.layers.Dense(units=2,activation='softmax')
    ])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])



history = model.fit(Norm_train_x,train_Y,epochs=25, validation_split=0.25)

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(history.history['loss'],'b-',label='loss')
plt.plot(history.history['val_loss'],'r--',label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(122)
plt.plot(history.history['accuracy'],'b-',label='acc')
plt.plot(history.history['val_accuracy'],'r--',label='val_acc')
plt.xlabel('Epoch')
plt.ylim(0.5,1)
plt.legend()

plt.show()
print(test_y.shape)
model.evaluate(Norm_test_x,test_Y)
a= model.predict(Norm_test_x)
pre_y = np.round(a[:,1])
print(test_y,pre_y)
comfu = []

class_names = ['0','1']
class_names = np.array(class_names)


from visual_matrix import plot_confusion_matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_y, pre_y)
np.set_printoptions(precision=1)
# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')
plt.show()
# from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,classification_report
    
# print(accuracy_score(test_y, pre_y))
# print(recall_score(test_y, pre_y))
# print(precision_score(test_y, pre_y))
# print(f1_score(test_y, pre_y))
