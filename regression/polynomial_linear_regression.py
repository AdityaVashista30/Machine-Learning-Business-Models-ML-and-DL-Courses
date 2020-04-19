# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 14:08:26 2020

@author: Hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

"""WE ARE NOT SPLITING DATA SET AS IT IS VERY SMALL i.e ONLY 10 OBSERVATIONS""""

#fiting linear model for comaparission
from sklearn.linear_model import LinearRegression
regressor_L=LinearRegression()
regressor_L.fit(x,y)

#fiting polynomial model
from sklearn.preprocessing import PolynomialFeatures
regressor_P=PolynomialFeatures(degree=4)
x_poly=regressor_P.fit_transform(x)
regressor_P.fit(x_poly,y)
regressor_L2=LinearRegression()
regressor_L2.fit(x_poly,y)

# Visualising the Linear Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor_L.predict(x), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(x, y, color = 'red')
plt.plot(x, regressor_L2.predict(regressor_P.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
x_grid=np.arange(min(x),max(x),0.1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor_L2.predict(regressor_P.fit_transform(x_grid.reshape(-1,1))), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()