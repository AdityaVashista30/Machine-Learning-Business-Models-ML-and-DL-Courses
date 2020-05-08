# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 19:52:49 2020

@author: Hp
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#STEP 1:DATA PREPROCESSING on training set
dataset_train=pd.read_csv("Google_Stock_Price_Train.csv")
training_set=dataset_train.iloc[:,1].values

#FEATURE SCALING
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler(feature_range=(0,1))
training_set_SC=sc.fit_transform(training_set.reshape(-1,1))

# Creating a data structure with 60 timesteps and 1 output
"""RNN WILL SEE 60 past info, then predict next output"""
x_train=[]
y_train=[]

for i in range(60,len(training_set_SC)):
    x_train.append(training_set_SC[i-60:i,0])
    y_train.append(training_set_SC[i,0])
    
x_train,y_train=np.array(x_train),np.array(y_train)

#RESHAPE
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

#ADDING 1st LSTM layer
regressor.add(LSTM(units=50,return_sequences = True,input_shape=(x_train.shape[1],1)))
#(no, of timestem, no of indicatores)
regressor.add(Dropout(rate=0.2))

#2nd Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a 3rd Layer
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a 4th Layer
regressor.add(LSTM(units = 50))#return_sequences = False
regressor.add(Dropout(0.2))

#Adding Output LAyer
regressor.add(Dense(units=1))

#COMPILING
regressor.compile(optimizer='adam',loss='mean_squared_error')
#FITTING
regressor.fit(x_train,y_train,batch_size=32,epochs=150)

#PREDICTION OF STOCK PRICE 2017
dataset_test=pd.read_csv("Google_Stock_Price_Test.csv")
real_stock_price=dataset_test.iloc[:,1].values

data_set_total=pd.concat((dataset_train['Open'],dataset_test['Open']),axis=0)#vertical concatination; adding lines

inputs=data_set_total[len(data_set_total)-len(dataset_test)-60:].values
inputs=inputs.reshape(-1,1)
inputs=sc.transform(inputs)

x_test=[]
for i in range(60,60+len(dataset_test)):
    x_test.append(inputs[i-60:i,:])
   
x_test=np.array(x_test)
x_test=np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

stock_price_predicted=regressor.predict(x_test)
stock_price_predicted=sc.inverse_transform(stock_price_predicted)

#PLOT AND COMAPRE

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(stock_price_predicted, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


