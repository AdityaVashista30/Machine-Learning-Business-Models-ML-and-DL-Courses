# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 19:00:14 2020

@author: Hp
"""

import pandas as pd
import numpy as np
import tensorflow 
from tensorflow import keras

 
dataset=pd.read_csv('Churn_Modelling.csv')
x = dataset.iloc[:, 3:14].values
y = dataset.iloc[:,-1].values

#encoding data of Gender
from sklearn.preprocessing import LabelEncoder
encoder_X2=LabelEncoder()
x[:,2]=encoder_X2.fit_transform(x[:,2])
#encoding data of country
from sklearn.preprocessing import OneHotEncoder
encoder_X1=LabelEncoder()
x[:,1]=encoder_X1.fit_transform(x[:,1])
hotEncoder=OneHotEncoder(categorical_features = [1])
x=hotEncoder.fit_transform(x).toarray()
x=x[:,1:]

#spliting in train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)

#standardisation
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

####PART 2: ANN BUILDING: A classifier
ann=tensorflow.keras.models.Sequential() 
#ADDING INPUT LAYER; 1st hidden layer
ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
#ADDING 2nd hidden layer
ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu")) 
#ADDING OUTPUT AND FINAL LAYER
"""SINCE WE ARE MAKING A BINARY CLASSIFIER, WE WILL BE HAVING ONW OUTPUT VALUE
i.e units=1 and we will be sigmoid activation instead of rectifier fucntion(most common) used above"""
ann.add(tensorflow.keras.layers.Dense(units=1,activation="sigmoid"))
""""IMPORTANT: if we are making multi class classifier then:-
units =n (number of classes; n>2)
activation= softmax

The softmax activation function is used in neural networks when we want to build a multi-class classifier which
 solves the problem of assigning an instance to one class when the number of possible classes is larger than two"""
 
 # Compiling the ANN
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
""""
optimizer: algo used to find optimized weights (stastic gradient- several available; most common=adam);
loss= loss function used to adjust weights; for more than 2 categories: use categorical_crossentropy;
metrics= criteria to evaluate model i.e rmse,accuracy etc"""


# Fitting the ANN to the Training set
ann.fit(x_train,y_train,batch_size = 32, epochs = 100)

""" 
batch size= number observation after which weights are updated
epochs=no of rounds"""
 

# Predicting the Test set results
y_pred = ann.predict(x_test)#floatin points in [0,1]
y_pred = (y_pred > 0.5)#convets numbers obtained into bool

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
