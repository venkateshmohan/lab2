# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 16:11:28 2018

@author: Venkatesh T Mohan
"""

import numpy as np
import pandas as pd
import sklearn
import time
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from keras.optimizers import SGD,adam,Adamax
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import confusion_matrix
# Function to create model, required for KerasClassifier
input_units=64
hidden_units=16
output_units=10

#Output activation units are softmax for all the models
#Cross-entropy model with relu as hidden activation units and optimizer adam
def model_sgd_entropy_relu():
	# create model
   model = Sequential()
   model.add(Dense(16, input_dim=input_units, activation='relu'))
   model.add(Dense(hidden_units, activation='relu'))
   model.add(Dense(output_units, activation='softmax'))
   #optimizer = SGD(lr=learn_rate,momentum=momentum)
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])   
   return model

#Cross-entropy model with tanh as hidden activation units and optimizer Adamax
def model_sgd_entropy_tanh(learn_rate=0.001):
   model1 = Sequential()
   model1.add(Dense(48, input_dim=input_units, activation='tanh'))
   model1.add(Dense(16, activation='tanh'))
   model1.add(Dense(output_units, activation='softmax'))
   #optimizer = SGD(lr=learn_rate,momentum=momentum)
   optimizer=Adamax(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
   model1.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])   
   return model1

#Mean-square error model with relu as hidden activation units and optimizer adam
def model_sgd_mse_relu(loss_fn='mean_squared_error',act='relu'):
	# create model
   model2 = Sequential()
   model2.add(Dense(32, input_dim=input_units, activation=act))
   model2.add(Dense(16, activation=act))
   model2.add(Dense(output_units, activation='softmax'))
   #optimizer=Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
   model2.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])   
   return model2

#Mean-square error model with tanh as hidden activation units and optimizer Adamax
def model_sgd_mse_tanh(learn_rate=0.002,bt_1=0.9,bt_2=0.999):#loss_fn='mean_squared_error',act='tanh',learn_rate=0.002,bt_1=0.9,bt_2=0.999
	# create model
   model3 = Sequential()
   model3.add(Dense(32, input_dim=input_units, activation='tanh'))
   model3.add(Dense(16, activation='tanh'))
   model3.add(Dense(output_units, activation='softmax'))
   optimizer=Adamax(lr=learn_rate, beta_1=bt_1, beta_2=bt_2, epsilon=None, decay=0.0)
   model3.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])   
   return model3

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# load dataset
train_ds = pd.read_csv("optdigits.csv", delimiter=",")
test_ds=pd.read_csv("test_optdigits.csv",delimiter=",")

# split into input  and output variables
train_X =train_ds.iloc[:, 0:64].values
train_Y =train_ds.iloc[:, 64].values
test_X =test_ds.iloc[:, 0:64].values
test_Y =test_ds.iloc[:, 64].values

#Input scaling alias normalization

def get_normed_mean_cov(X):
    standard = StandardScaler().fit_transform(X)
    mean_X = np.mean(standard, axis=0)  
    X_cov = (standard - mean_X).T.dot((standard - mean_X)) / (standard.shape[0]-1)  
    return standard, mean_X, X_cov

train_X, _, _ = get_normed_mean_cov(train_X)
test_X, _, _ = get_normed_mean_cov(test_X)

#Encoding output values of train and test to categorical values

def one_to_c_encodingtrain(train_Y):
    encoding_y = np_utils.to_categorical(train_Y)
    return encoding_y

def one_to_c_encodingtest(test_Y):
    test_encoding_y=np_utils.to_categorical(test_Y)
    return test_encoding_y
# create model
model = KerasClassifier(build_fn=model_sgd_entropy_relu, epochs=30, batch_size=32, verbose=1)
model1 = KerasClassifier(build_fn=model_sgd_entropy_tanh, epochs=30, batch_size=32, verbose=1)
model2 = KerasClassifier(build_fn=model_sgd_mse_relu, epochs=30, batch_size=32, verbose=1)
model3 = KerasClassifier(build_fn=model_sgd_mse_tanh, epochs=30, batch_size=32, verbose=1)

# define early stopping callback
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, \
                          verbose=1, mode='auto')

# train the model
def model_entropy_relu_train(train_X,encoding_y):
    start=time.time()
    model.fit(train_X,encoding_y,callbacks=[earlystop],validation_split=0.2)
    #print(his.history['val_acc'])
    end = time.time()
    return end-start


#Calculate the overall accuracy of the model by doing an average over all accuracies of each iterations
def model_entropy_relu_scores():
    scores = model.score(test_X, one_to_c_encodingtest(test_Y))
    return scores*100

print("Model took %0.2f seconds to train and the overall accuracy is"%(model_entropy_relu_train(train_X,one_to_c_encodingtrain(train_Y))),model_entropy_relu_scores())

#Fitting the model and calculating convergence speed of the training model

def model_entropy_tanh_train(train_X,encoding_y):
    start1=time.time()
    model1.fit(train_X,encoding_y,callbacks=[earlystop],validation_split=0.2)
    end1 = time.time()
    return end1-start1

def model_entropy_tanh_scores():
    scores = model1.score(test_X, one_to_c_encodingtest(test_Y))
    return scores*100

print("Model took %0.2f seconds to train and the overall accuracy is"%(model_entropy_tanh_train(train_X,one_to_c_encodingtrain(train_Y))),model_entropy_tanh_scores())

def model_mse_relu_train(train_X,encoding_y):    
    start2=time.time()
    model2.fit(train_X,encoding_y,callbacks=[earlystop],validation_split=0.2)
    end2 = time.time()
    return end2-start2

def model_mse_relu_scores():
    scores = model2.score(test_X, one_to_c_encodingtest(test_Y))
    return scores*100

print("Model took %0.2f seconds to train and the overall accuracy is"%(model_mse_relu_train(train_X,one_to_c_encodingtrain(train_Y))),model_mse_relu_scores())

def model_mse_tanh_train(train_X,encoding_y):
    start3=time.time()
    model3.fit(train_X,encoding_y,callbacks=[earlystop],validation_split=0.2)
    end3 = time.time()
    return end3-start3

def model_mse_tanh_scores():
    scores = model3.score(test_X, one_to_c_encodingtest(test_Y))
    return scores*100

print("Model took %0.2f seconds to train and the overall accuracy is"%(model_mse_tanh_train(train_X,one_to_c_encodingtrain(train_Y))),model_mse_tanh_scores())


#Predict the model
x_pred=model.predict(train_X)
y_pred = model.predict(test_X)

#Calculation of confusion matrix and classification reports for all the models
cm_train = confusion_matrix(train_Y, x_pred)
print(cm_train)
cm_test = confusion_matrix(test_Y, y_pred)
print(cm_test)

print(classification_report(test_Y,y_pred))

    
x_pred1 = model1.predict(train_X)
y_pred1 = model1.predict(test_X)

cm_train1 = confusion_matrix(train_Y, x_pred1)
print(cm_train1)
cm_test1 = confusion_matrix(test_Y, y_pred1)
print(cm_test1)

print(classification_report(test_Y,y_pred1))
    
x_pred2 = model2.predict(train_X)
y_pred2 = model2.predict(test_X)

cm_train2 = confusion_matrix(train_Y, x_pred2)
print(cm_train2)
cm_test2 = confusion_matrix(test_Y, y_pred2)
print(cm_test2)

print(classification_report(test_Y,y_pred2))

x_pred3 = model3.predict(train_X)
y_pred3 = model3.predict(test_X)

cm_train3 = confusion_matrix(train_Y, x_pred3)
print(cm_train3)
cm_test3 = confusion_matrix(test_Y, y_pred3)
print(cm_test3)

print(classification_report(test_Y,y_pred3))
