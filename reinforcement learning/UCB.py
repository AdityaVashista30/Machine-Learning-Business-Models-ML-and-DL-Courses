# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 04:06:24 2020

@author: Hp
"""

import matplotlib.pyplot as plt
import pandas as pd
import math

dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing UCB from scratch
N=10000
d=10

number_of_selections=[0]*d
sum_of_rewards=[0]*d
ads_selected=[]
total_reward=0

for n in range(0,N):
    max_upper_bound=0
    ad=0
    for i in range(0,d):
        if (number_of_selections[i] > 0):
            average_reward=sum_of_rewards[i]/number_of_selections[i]
            delta_i=math.sqrt((3*math.log(n+1))/(2*number_of_selections[i]))
            upper_bound=average_reward + delta_i
        else:
            upper_bound=1e400
        if upper_bound>max_upper_bound:
            max_upper_bound=upper_bound
            ad=i
    ads_selected.append(ad)
    number_of_selections[ad]+=1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

#Visualize
plt.hist(ads_selected)
plt.title("HISTOGRAM OF SELECTED ADS")
plt.xlabel("Ad")
plt.ylabel("No. of Selections")
plt.show()