# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 13:50:57 2018

@author: Akhtar
"""

# Convolutional Neural Network Resnet Architecture

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
import numpy as np
np.random.seed(1337)
import pandas as pd
import os
from sklearn import preprocessing,cross_validation,neighbors
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Flatten

from keras.layers import Input
from keras.layers import Dense
from keras.optimizers import RMSprop
from keras.layers.convolutional import Conv2D
from keras.layers import merge
from sklearn.metrics import confusion_matrix
from keras.layers.pooling import MaxPooling2D
from keras.utils import np_utils
import keras
batch_size = 128
nb_classes = 2
nb_epoch = 20
dfs1=[]

# Define Path for DataSet

Positive_Class='C:\\Users\\Akhtar\\Documents\\Python Scripts\\DataSets\\Positive_Class\\'

Negative_Class='C:\\Users\\Akhtar\\Documents\\Python Scripts\\DataSets\\Negative_Class\\'

df3=pd.DataFrame({'Address':[Negative_Class],
                  'Label': [0]})
#list the files

for item in os.listdir(Negative_Class):
   
    concat_string = Negative_Class+item
    
        
    df5=pd.DataFrame({'Address':[concat_string],
                          'Label': [0]})

    df3=df3.append(df5)
       


######

for item in os.listdir(Positive_Class):
   
    concat_string = Positive_Class+item
        
    df5=pd.DataFrame({'Address':[concat_string],
                      'Label': [1]})
    df3=df3.append(df5)
    Big_matrix = []

k=0
label=[]
for index, row in df3.iterrows():
  
    if k!=0:
        label.append(row['Label'])
        df=pd.read_csv(row['Address'],header=None,sep=',')
        
        aray=df.as_matrix()
        
        xp=np.reshape(aray,(100,100,3))
        matrix=xp
        
        Big_matrix.append(matrix)

    k += 1
    
    LAbel_array=np.array(label)

x=np.array(Big_matrix)
#y=LAbel_array.reshape(LAbel_array.shape[0],1)
y = LAbel_array

x_train,x_test,y_train,y_test=cross_validation.train_test_split(x,y,test_size=0.2)

#because we have float value
x_train = x_train.astype('float32')
x_test=x_test.astype ('float32')

#here we normalize the data 0-1
#x_train /= 255
#x_test /= 255

print(x_train.shape)
print(x_train.shape[0],'train sample')
print(x_test.shape[0],'test sample')
# 1 hot representation ( 1 hot means from all values only one is all other element is zeo make all vector
#for all label)

y_true = y_test
y_train=np_utils.to_categorical(y_train,nb_classes)
y_test=np_utils.to_categorical(y_test,nb_classes)
visible = Input(shape=(100,100,3))
conv1 = Conv2D(3, (3,3),padding='same')(visible)
s = keras.layers.add([visible,conv1])
pool1 = MaxPooling2D(pool_size=(2, 2))(s)
conv2 = Conv2D(16, kernel_size=4, activation='tanh')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

hidden1 = Dense(10, activation='relu')(pool2)
hidden2 = Flatten()(hidden1)
output = Dense(2, activation='softmax')(hidden2)

model = Model(inputs=visible, outputs=output)
# summarize layers
model.summary()

rms = RMSprop()
model.compile(optimizer=rms, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_data = (x_test, y_test))
score = model.evaluate(x_test,y_test, verbose=0)
print('test score:', score[0])
print('test accuracy:',score[1])
y_predict = np.argmax(model.predict(x_test), -1)
print("Confusion matrix is :")
print(confusion_matrix(y_true, y_predict))

from sklearn.metrics import f1_score
print("F1 Score is :")
print(f1_score(y_true, y_predict))
