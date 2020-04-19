# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 00:42:16 2020

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

#Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
#STEP1: Making list of dictionaries of parmeters
parameters1=[{'C':[1,10,100,1000],'kernel':['linear']},
             {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.2,0.01,0.001,0.0001]},
             {'C':[1,10,100,1000],'kernel':['poly'],'degree':[2,3,4,5,6]}]

best=[]
#STEP 2: FITING AND RESULT
grid_search1=GridSearchCV(estimator=classifier,param_grid=parameters1,scoring='accuracy',n_jobs=-1,cv=15)
grid_search1.fit(x_train,y_train)
best_accuracy1 = grid_search1.best_score_
best_parameters1 = grid_search1.best_params_
best.append([best_accuracy1,best_parameters1])

parameters2=[{'criterion':['entropy'],'splitter':['best','random']},
             {'criterion':['gini'],'splitter':['best','random']}]
grid_search2=GridSearchCV(estimator=class1,param_grid=parameters2,scoring='accuracy',n_jobs=-1,cv=15)
grid_search2.fit(x_train,y_train)
best_accuracy2 = grid_search2.best_score_
best_parameters2 = grid_search2.best_params_
best.append([best_accuracy2,best_parameters2])

parameters3=[{'n_estimators':[10,15,20,25,29,30,40,50,100],'criterion':['entropy']},
             {'n_estimators':[10,15,20,25,29,30,40,50,100],'criterion':['gini']}]
grid_search3=GridSearchCV(estimator=class2,param_grid=parameters3,scoring='accuracy',n_jobs=-1,cv=15)
grid_search3.fit(x_train,y_train)
best_accuracy3 = grid_search3.best_score_
best_parameters3 = grid_search3.best_params_
best.append([best_accuracy3,best_parameters3])

print(best)


