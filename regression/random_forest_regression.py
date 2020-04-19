# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:41:55 2020

@author: Hp
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators = 500, random_state = 0)#n_estimator=no of trees
regressor.fit(x,y)
# Predicting a new result
regressor.predict(np.array([6.5]).reshape(1,-1))


# Visualising the Decision Tree Regression results (higher resolution)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color = 'red')
plt.scatter(8,regressor.predict(np.array([8]).reshape(1,-1)), color = 'yellow')
plt.scatter(6.5,regressor.predict(np.array([6.5]).reshape(1,-1)), color = 'yellow')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


print(regressor.predict(np.array([6.5]).reshape(1,-1)))