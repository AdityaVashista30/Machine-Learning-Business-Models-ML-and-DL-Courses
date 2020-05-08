# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 01:22:02 2020

@author: Hp
"""

#MOVIE RECOMANDATION SYSTEM 

import pandas as pd
import numpy as np
import torch
import torch.nn 
import torch.nn.parallel as parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

#Importing DataSet
movies=pd.read_csv('ml-1m/movies.dat',sep='::',header=None,engine='python', encoding = 'latin-1')
users=pd.read_csv('ml-1m/users.dat',sep='::',header=None,engine='python', encoding = 'latin-1')
ratings=pd.read_csv('ml-1m/ratings.dat',sep='::',header=None,engine='python', encoding = 'latin-1')

#prepairing trainingg and test set
training_set=pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set=np.array(training_set,dtype='int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')


d=['u2','u3','u4','u5']

for i in d:
    str1='ml-100k/'+i+'.base'
    str2='ml-100k/'+i+'.test'
    temp1=pd.read_csv(str1, delimiter = '\t')
    temp2=pd.read_csv(str2, delimiter = '\t')
    temp1=np.array(temp1,dtype='int')
    temp2=np.array(temp2,dtype='int')
    training_set = np.concatenate((training_set,temp1),axis=0)
    test_set = np.concatenate((test_set,temp2),axis=0)

#Getting number of users and movies
nb_users=int(max(max(training_set[:,0]), max(test_set[:,0])))
nb_movies=int(max(max(training_set[:,1]), max(test_set[:,1])))

# Converting the data into an array with users in rows and movies in columns
def convert(data):
    new_data=[]
    for id_user in range(1,nb_users+1):
        id_movies=data[:,1][data[:,0]==id_user]
        id_ratings=data[:,2][data[:,0]==id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set=convert(training_set)
test_set=convert(test_set)

# Converting the data into Torch tensors
training_set=torch.FloatTensor(training_set)
test_set=torch.FloatTensor(test_set)   
        
#Converting the ratings into binary ratings 1 (Liked) or 0 (Not Liked);-1(not reviwed)
#Key: Liked-rating >=3; else not liked
training_set[training_set==0]= -1
training_set[training_set==1]=0
training_set[training_set==2]=0
training_set[training_set>=3]=1

test_set[test_set==0]= -1
test_set[test_set==1]=0
test_set[test_set==2]=0
test_set[test_set>=3]=1


# Creating the architecture of the Neural Network
class RBM:
    def __init__(self,nv,nh):#nv=number of visible nodes; nh=number of hidden nodes
        self.W=torch.randn(nh,nv)  #weights; initialized in a matrix of (nh X nv)
        self.a=torch.randn(1,nh)  #bias for hidden nodes; vector of size nh
        self.b=torch.randn(1,nv)  #bias for visible nodes; vector of size nv
    
    def sample_h(self,x): #x will correspond to the visible neurons, v in the probabilities, p h given v.
    #The second function is about sampling the hidden nodes according to the probabilities, p h given v 
    #where h is a hidden node and v is a visible node 
        wx=torch.mm(x,self.W.t()) #product of 2 tensor objects
        activation= wx + self.a.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    
    def sample_v(self,y): #y will correspond to the hidden neurons, h in the probabilities, p v given h
        #same as above function; for visible nodes
        wy=torch.mm(y,self.W) #product of 2 tensor objects
        activation= wy + self.b.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    
    def train(self, v0, vk, ph0, phk):
    #v0=our input vector containing the ratings of all the movies by one user
    #vk= that's the visible nodes obtained after K samplings. You know after K round trips from the visible nodes to hidden
    #ph0=  the vector of probabilities that at the first iteration the hidden nodes equal one given the values of V zero
    #phk= that will correspond to the probabilities of the hidden nodes after K sampling given the values of the visible nodes, VK.
        self.W+=(torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b+=torch.sum((v0 - vk), 0)
        self.a+=torch.sum((ph0 - phk), 0)
     
    def predict( self, x): # x: visible nodes
        _, h = self.sample_h( x)
        _, v = self.sample_v( h)
        return v
    
    def compileModel(self,training_set,epochs,batch_size,nb_users):
        for epoch in range(1,epochs+1):#implement epochs
            training_loss=0
            c=0.0  #counter
            #training in batches
            for id_user in range(0,nb_users-batch_size,batch_size):
                vk=training_set[id_user:id_user+batch_size]
                v0 = training_set[id_user:id_user+batch_size]
                ph0,_=self.sample_h(v0)
                #now doing k sampling
                for i in range(10):
                    _,hk = self.sample_h(vk)
                    _,vk = self.sample_v(hk)
                    vk[v0<0] = v0[v0<0]
                phk,_=self.sample_h(vk)
                self.train(v0, vk, ph0, phk)
                training_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
                c += 1.
        print('epoch: '+str(epoch)+' loss: '+str(training_loss/c))

nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

#Training RBM:
epochs= 15
for epoch in range(1,epochs+1):#implement epochs
    training_loss=0
    c=0.0  #counter
    #training in batches
    for id_user in range(0,nb_users-batch_size,batch_size):
        vk=training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0,_=rbm.sample_h(v0)
        #now doing k sampling
        for i in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0<0] = v0[v0<0]
        phk,_=rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        training_loss += torch.mean(torch.abs(v0[v0>=0] - vk[v0>=0]))
        c += 1.
    print('epoch: '+str(epoch)+' loss: '+str(training_loss/c))
    
    

# Testing the RBM
test_loss = 0
s = 0.
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt>=0]) > 0:
        _,h = rbm.sample_h(v)
        _,v = rbm.sample_v(h) 
        test_loss += torch.mean(torch.abs(vt[vt>=0] - v[vt>=0]))
        s += 1.
print('test loss: '+str(test_loss/s))

user_IN=int(input("ENTER USER ID (Only number/int): "))

movie=str(input("ENTER Movie Name : "))

for i in range( len(movies)):    
    if movies[1][i]==movie:
        movie_IN=i
        break
    else:
        continue
    
p=rbm.predict(training_set[user_IN-1:user_IN])
if p[0][movie_IN]==1:
    print("User ",user_IN," WILL LIKE ",movie)
else:
    print("User ",user_IN," WILL NOT LIKE ",movie)