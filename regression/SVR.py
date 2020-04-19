# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:51:19 2020

@author: Hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv("Position_Salaries.csv")
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

#no test case spliting as we have only 10 observations

#FEATURE SCALING IS IMPORTANT HERE AS SVR DOESN'T APPLY IT ITSELF
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x= sc_x.fit_transform(x.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))  

# Fitting SVR to the dataset
from sklearn.svm import SVR
regressor=SVR()
regressor.fit(x,y)

#NOW we apply transform again to get proper value
y_pred=regressor.predict(sc_x.transform(np.array([3.5]).reshape(-1,1)))
y_pred = sc_y.inverse_transform(y_pred)


# Visualising the SVR results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
x_grid = np.arange(min(x), max(x), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()