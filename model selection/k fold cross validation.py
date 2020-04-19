# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 22:30:56 2020

@author: Hp
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)
y_p1 = classifier.predict(x_test)


from sklearn.tree import DecisionTreeClassifier
class1=DecisionTreeClassifier(criterion="entropy",random_state=0)
class1.fit(x_train,y_train)
yP1=class1.predict(x_test)

from sklearn.ensemble import RandomForestClassifier
class2=RandomForestClassifier(n_estimators=29,criterion="entropy",random_state=0)
class2.fit(x_train,y_train)
yP2=class2.predict(x_test)

from sklearn.linear_model import LogisticRegression
class3=LogisticRegression(random_state=0)
class3.fit(x_train,y_train)
yP3=class3.predict(x_test)

acc_table=[]
#COMPUTING PREDICTING POWER BY K Fold Cross Validation
from sklearn.model_selection import cross_val_score as KFCV
acc1=KFCV(estimator=classifier,X=x_train,y=y_train,cv=15,n_jobs=-1)
acc_table.append(["svm",acc1.mean(),acc1.std()])

acc2=KFCV(estimator=class1,X=x_train,y=y_train,cv=15,n_jobs=-1)
acc_table.append(["Decision Tree",acc2.mean(),acc2.std()])


acc3=KFCV(estimator=class2,X=x_train,y=y_train,cv=15,n_jobs=-1)
acc_table.append(["Random Forest",acc3.mean(),acc3.std()])


acc4=KFCV(estimator=class3,X=x_train,y=y_train,cv=15,n_jobs=-1)
acc_table.append(["Logistic",acc4.mean(),acc4.std()])

print(np.matrix(acc_table))




