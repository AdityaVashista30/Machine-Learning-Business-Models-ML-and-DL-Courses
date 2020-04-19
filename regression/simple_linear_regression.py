# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:01:05 2020

@author: Hp
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('Salary_Data.csv')
x=dataset.iloc[:,0].values
y=dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state= 1)

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression() 
regressor.fit(x_train.reshape(-1,1),y_train)
#predict data
y_predict=regressor.predict(x_test.reshape(-1,1))

# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'yellow')
plt.plot(x_train, regressor.predict(x_train.reshape(-1,1)), color = 'blue')#regression_line
plt.scatter(x_test,y_test,color='green')
plt.scatter(x_test,y_predict,color='red')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title("LINEAR REGRESSION MODEL")
plt.show()
