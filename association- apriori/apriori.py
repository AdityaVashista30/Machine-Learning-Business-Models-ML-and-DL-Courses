# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 12:51:12 2020

@author: Hp
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
#converting all items in dataset to string
df=dataset.applymap(str)

#converting dataset into list of list, where nested lists are all single transaction
trans=[]
for i in range(len(dataset)):
    trans.append((df.iloc[i,:].values).tolist())
    
from apyori import apriori
rules=apriori(trans,min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)


# Visualising the results
results = list(rules)

#displaying rules:
for i in range(len(results)):
    items=results[i][0]
    print("Rule ",i+1,end=" ")
    for j in items:
        print("->",j,end=" ")
    print(" ")