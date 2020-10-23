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
from visual_matrix import plot_confusion_matrix
import itertools
# print(os.listdir('C:/Users/pjy/covid_analysis/val/'))

# #데이터 수량
# dATA_N = array_data(normal_N)
# dATA_P =array_data(normal_P)
# dATA_N, dATA_P = np.array(dATA_N,dtype=float), np.array(dATA_P,dtype=float)
# print("정상데이터:\t", dATA_N.shape[0])
# print("환자데이터:\t",dATA_P.shape[0])

## 데이터 준비
train_norm = 'C:/Users/pjy/covid_analysis/train/train_norm.csv' 
train_PNE= 'C:/Users/pjy/covid_analysis/train/PNE.csv' 
test_norm = 'C:/Users/pjy/covid_analysis/test/test_norm.csv' 
test_PNE= 'C:/Users/pjy/covid_analysis/test/test_PNE.csv' 
val_norm = 'C:/Users/pjy/covid_analysis/val/val_norm.csv' 
val_PNE ='C:/Users/pjy/covid_analysis/val/val_PNE.csv' 
testSet_norm = 'C:/Users/pjy/covid_analysis/total_test/normal_test.csv'
testSet_PNE = 'C:/Users/pjy/covid_analysis/total_test/covid_test.csv' 
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
Test_N,Test_P = csv_load(testSet_norm, 0), csv_load(testSet_PNE, 1)
#데이터 셋 만들기
normal_N = pd.concat([train_N,test_N,val_N])
normal_P = pd.concat([train_P,test_P,val_P])
covid_X = pd.concat([normal_N,normal_P])
covid_test = pd.concat([Test_N,Test_P])
covid_X.to_csv('C:/Users/pjy/covid_analysis/covid_19.csv')
covid_test.to_csv('C:/Users/pjy/covid_analysis/total_test/covid_19(T).csv')

#데이터 섞기
covid_X_shuffle = covid_X.sample(frac=1).reset_index(drop=True)
covid_test_shuffle = covid_test.sample(frac=1).reset_index(drop=True)

# print(covid_X_shuffle)
covid_np = array_data(covid_X_shuffle)
covid_np_T = array_data(covid_test_shuffle)
# print(covid_np.shape)
# print(covid_np_T.shape)

# # sampling imag
# for c in range(10,21,2):
    
#     covid_1 = covid_np_T[c]
#     covid_1 ,covid_label = covid_1[:-1],covid_1[-1]

#     covid_1 = covid_1.reshape(128,128)
#     covid_1 = np.array(covid_1,dtype=float)
#     plt.title(covid_label)
#     plt.imshow(covid_1,'gray')
#     plt.colorbar()
#     plt.show()
    
data_x ,data_ys = covid_np[:,:-1],covid_np[:,-1]
data_x_T, data_ys_T = covid_np_T[:,:-1],covid_np_T[:,-1]
data_y = np.array(data_ys, dtype=int)
data_y_T = np.array(data_ys_T, dtype=int)
# #정규화
data_x  = data_x/255.0
data_x_T = data_x_T/255.0
Data_x =np.array(data_x,dtype=float)
Data_x_T = np.array(data_x_T, dtype=float)
# #모델에 적용 시키기 위하여 reshape(batch_size,height,width,channel)
input_x= Data_x.reshape(5986,128,128,1)
input_x_T = Data_x_T.reshape(2003,128,128,1)
data_y = tf.keras.utils.to_categorical(data_y,num_classes=2)
data_y_T = tf.keras.utils.to_categorical(data_y_T,num_classes=2)

input_list = list(input_x)
data_Y = list(data_y)
data_ys = list(data_ys)
data_ys_T1 = list(data_ys_T)
data_ys_T1 = np.array(dat_ys_T)
k=5
num_validation = len(input_list) // k
num_label = len(data_Y)//k

validation_scores = []
class_names = ['0','1']
class_names = np.array(class_names)

for fold in range(k):
    print("{}번째 훈련 중 입니다!!!!!".format(fold+1))
    validation_data = input_list[num_validation * fold: num_validation * (fold + 1)]
    train_data = input_list[:(num_validation * fold) ]+input_list[num_validation *(fold+1):] 
    
    validation_label = data_Y[num_label * fold: num_label * (fold + 1)]
    train_label = data_Y[:(num_label * fold) ]+data_Y[num_label *(fold+1):] 
    
    yd_test =data_ys[num_label * fold: num_label * (fold + 1)]
    yd_train = data_ys[:(num_label * fold) ]+data_ys[num_label *(fold+1):]
    
    test_data , test_label,yd_test= np.array(validation_data),np.array(validation_label),np.array(yd_test)
    train_x_data,train_y_data,yd_train = np.array(train_data), np.array(train_label),np.array(yd_train)
   
# # 모델구축
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

    model.summary()

    history = model.fit(train_x_data,train_y_data,epochs=25)
    val_score = model.evaluate(test_data,test_label)
    validation_scores.append(val_score)
    
    #결과 시각화
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(history.history['loss'],'b-',label='loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(122)
    plt.plot(history.history['accuracy'],'b-',label='acc')
    plt.xlabel('Epoch')
    plt.ylim(0.5,1)
    plt.legend()

    plt.show()
    
    ## 예측값 추출
    predic_test= model.predict(test_data)
    predic_train= model.predict(train_x_data)
    pre_test_y = np.round(predic_test[:,1])
    pre_train_y =np.round(predic_train[:,1])
    
    cnf_matrix_train = confusion_matrix(np.array(yd_train,dtype=float), pre_train_y)
    cnf_matrix_test = confusion_matrix(np.array(yd_test,dtype=float), pre_test_y)
    np.set_printoptions(precision=2)
    
    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=class_names,
                      title='Confusion matrix, train {}'.format(fold))
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_train, classes=class_names, normalize=True,
                      title='Normalized confusion matrix,train {}'.format(fold))
    plt.figure()
    plot_confusion_matrix(cnf_matrix_test, classes=class_names,
                      title='Confusion matrix, test {}'.format(fold))
    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix_test, classes=class_names, normalize=True,
                      title='Normalized confusion matrix,test {}'.format(fold))
    plt.show()


validation_score = np.array(validation_scores)
val = np.average(validation_score[:,1])

print("검증데이터 결과(=accuracy):", validation_score[:5,1])
print("평균 검증데이터 결과(=accuracy): {}".format( val))


total_train= np.array([[1,0],[0,1]])
plt.figure()
plot_confusion_matrix(total_train, classes=class_names, normalize=True,
                      title='Normalized confusion matrix,train_mean')
plt.show()

test_score = model.evaluate(input_x_T,data_y_T)
print(test_score)

test_predic = model.predict(input_x_T)
test_predict = np.round(test_predic[:,1])
test_confuse = confusion_matrix(np.array(data_ys_T1,dtype=float), test_predict)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(test_confuse, classes=class_names, normalize=True,
                      title='TEST_SET')
plt.show()
    
print("Test_set Accuracy:",test_score[1])