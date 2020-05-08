# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 06:10:49 2020

@author: Hp
"""

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
x = dataset.iloc[:, 3:13].values
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

#ADDING Dropout at value rate=0.1 or 10%
"""This is done at multiple layers of neural network to avoid overfiting and get better predictions, 
for high volume of data and multiple layers, it should bea llowed at all layers
rate=x means x fraction/percent of features to be disabled """
ann.add(tensorflow.keras.layers.Dropout(rate=0.2))

#ADDING 2nd hidden layer
ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu")) 
ann.add(tensorflow.keras.layers.Dropout(rate=0.2))
 
#Adding 3rd hidden layer
ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu")) 
ann.add(tensorflow.keras.layers.Dropout(rate=0.2))
 

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
ann.fit(x_train,y_train,batch_size = 32, epochs = 200)

""" 
batch size= number observation after which weights are updated
epochs=no of rounds"""
 

# Predicting the Test set results
y_pred = ann.predict(x_test)#floatin points in [0,1]
y_pred = (y_pred > 0.5)#convets numbers obtained into bool

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print((cm[0][0]+cm[1][1])/2000)

"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
newP=ann.predict(sc.transform(np.array([0,0,600,1,40,3,60000,2,1,1,50000]).reshape(1,-1)))
newP=newP>0.5

#EVALUATION OF MODEl
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score 
#create a function
def build_ann_classifier(): 
    #COPY AND PASTE THE ALL THE STEPS OF BUILDING AN ANN
    ann=tensorflow.keras.models.Sequential() 
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu")) 
    ann.add(tensorflow.keras.layers.Dense(units=1,activation="sigmoid"))
    ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann

#APPLYING K-FOLDS
classifier=KerasClassifier(build_fn=build_ann_classifier,batch_size=32,epochs=200)
accuracy=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,n_jobs=1)    
print(accuracy.mean()," ",accuracy.std())

#Grid Search
from sklearn.model_selection import GridSearchCV

def build_ann_classifier2(optimizer,metrics): 
    #COPY AND PASTE THE ALL THE STEPS OF BUILDING AN ANN
    ann=tensorflow.keras.models.Sequential() 
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu")) 
    ann.add(tensorflow.keras.layers.Dense(units=1,activation="sigmoid"))
    ann.compile(optimizer=optimizer, loss = 'binary_crossentropy', metrics = metrics)
    return ann
parameters=[{'optimizer':['adam','rmsprop','adamax'],'epochs':[100,200,300,400,500],'batch_size':[25,32,20],
             'metrics':[['accuracy'],['binary_accuracy']]}]

classifier2 = KerasClassifier(build_fn = build_ann_classifier2)
gridS=GridSearchCV(estimator = classifier2,param_grid = parameters,scoring = 'accuracy',cv = 10)
gridS.fit(x_train, y_train)   
best_parameters = gridS.best_params_
best_accuracy = gridS .best_score_
print(best_parameters,best_accuracy)
"""{'batch_size': 32, 'epochs': 300, 'metrics': ['accuracy'], 'optimizer': 'adamax'}, 0.86325"""

#Fitting best parameters
def build_best_ann_classifier(): 
    #COPY AND PASTE THE ALL THE STEPS OF BUILDING AN ANN
    ann=tensorflow.keras.models.Sequential() 
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=6,activation="relu"))
    ann.add(tensorflow.keras.layers.Dense(units=1,activation="sigmoid"))
    ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return ann

#APPLYING K-FOLDS on best params
classifier=KerasClassifier(build_fn=build_best_ann_classifier,batch_size=32,epochs=300)
accuracy=cross_val_score(estimator=classifier,X=x_train,y=y_train,cv=10,n_jobs=1)    
print(accuracy.mean()," ",accuracy.std())