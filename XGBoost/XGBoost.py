# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 02:38:00 2020

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

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
