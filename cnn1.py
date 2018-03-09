# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 02:33:16 2018

@author: Venkatesh T Mohan
"""
import numpy as np
import pandas as pd
import time
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from keras.callbacks import EarlyStopping
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


#Load train and test dataset
train_ds = pd.read_csv("optdigits.csv",delimiter=",")
train_X= train_ds.iloc[:,0:64].values
train_Y= train_ds.iloc[:,64].values

test_ds = pd.read_csv("test_optdigits.csv",delimiter=",")
test_X  = test_ds.iloc[:,0:64].values 
test_Y  = test_ds.iloc[:,64].values

#Reshape the model to 4 dimensions with dimensions (depth,convolutionmatrix_height,convolutionmatrix_breadth,rgb)
train_X = train_X.reshape(-1,8,8,1)
test_X = test_X.reshape(-1,8,8,1)


#Encoding output values of train and test data to categorical values
y_train = np_utils.to_categorical(train_Y)
y_test = np_utils.to_categorical(test_Y)



#Building CNN Model with batch size(number of epochs per update) as 128 
#number of epochs(iterations) is 30, filter window is chosen as 3x3 convolutional matrix
#the first half of image is given with number of filters as 32 and bottom half also 32 filters
#Max pooling is applied to convolutional layers with pool size as (2,2) and drop out probabilities with 0.3 and 0.5 to prevent overfitting
#Early stopping technique is the regularization technique used to avoid over-fitting with each iteration
epochs=30 
bs=128
output_units=10  
hidden_units=128
drop_out_1=0.3
drop_out_2=0.5
def create_cnn(epochs,bs): 
    model = Sequential()

    model.add(Convolution2D(32, 3, 3 , input_shape=(8,8,1), 
                        activation='relu',
                        border_mode = 'valid'))
    model.add(Convolution2D(32, 3, 3, border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2,2), dim_ordering='th')) 
    model.add(Dropout(drop_out_1))
    model.add(Flatten())
    model.add(Dense(hidden_units, activation='relu'))
    model.add(Dropout(drop_out_2))
    model.add(Dense(output_units, activation='softmax'))   
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
    earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5)
    start=time.time()
    model.fit(train_X,y_train, callbacks=[earlystop], batch_size=bs, nb_epoch=epochs, 
                validation_split=0.2)
    end=time.time()
    print("Model took seconds to train:",end-start)
    return model

#Calculate the overall accuracy of the model by doing an average over all accuracies of each iterations
model=create_cnn(epochs,bs)
def model_evaluate():
    scores = model.evaluate(test_X, y_test, verbose=0)
    return scores[1]*100
    
print(model_evaluate())
 
#Predict the model   
def model_predict():
    pred_y=model.predict_classes(test_X)
    pred_x=model.predict_classes(train_X)
    return pred_x,pred_y

pred_x,pred_y=model_predict()

#Calculation of confusion matrix and classification reports
from sklearn.metrics import confusion_matrix
def conf_matrix(train_Y,test_Y):
    cm_train = confusion_matrix(train_Y, pred_x)
    print(cm_train)
    cm_test = confusion_matrix(test_Y, pred_y)
    print(cm_test)
    print(classification_report(test_Y,pred_y))
    
conf_matrix(train_Y,test_Y)  
