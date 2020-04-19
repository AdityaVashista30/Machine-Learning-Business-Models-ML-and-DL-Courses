# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:54:38 2020

@author: Hp
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset=pd.read_csv('50_Startups.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#encoding categorial data; in this case colum=State
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder=LabelEncoder()
x[:,3]=labelEncoder.fit_transform(x[:,3])
oneHotEncoder=OneHotEncoder(categorical_features=[3])
x=oneHotEncoder.fit_transform(x).toarray()

# Avoiding the Dummy Variable Trap 
#i.e to have only n-1 dummy variables from n dummy variables generated from encoder
x = x[:, 1:]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)

"""
#building optimal model using backward elimination
import statsmodels.regression.linear_model as sm
#NECESSARY to add an array of ones in starting of dataset
x=np.append(arr=np.ones((50,1)).astype(int), values=x,axis=1)
#create optimal matrix
x_opt=x
#fitting model for backward elimination
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
#looking at summary of generated model
regressor_OLS.summary()
#remove the column with max P value, value>=0.05
x_opt=x[:,[0,1,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#again
x_opt=x[:,[0,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#again
x_opt=x[:,[0,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
#again
x_opt=x[:,[0,3]]
regressor_OLS=sm.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
"""
###AUTOMATIC BACKWARD ELIMINATION USING P values
import statsmodels.regression.linear_model as sm
x=np.append(arr=np.ones((50,1)).astype(int), values=x,axis=1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
      
    regressor_OLS.summary()
    return x
 

SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)



###AUTOMATIC BACKWARD ELIMINATION USING P values and R-squared
"""import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05"""