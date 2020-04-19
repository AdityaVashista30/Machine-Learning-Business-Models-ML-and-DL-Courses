# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:53:12 2020

@author: Hp
"""

import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter="\t",quoting=3)

"""CLEAN THE TEXT: TO CREATE BAG OF USEFUL WORDS, WHICH CONSISTS WORDS USED TO DETERMINE THE TYPE
STEMMING, REMOVE UNNECESARY WORDS, PUNCTUATION"""
"""BASIC KNOWLEDGE USED FOR EACH REVIEW:
import re 
#STEP 1: KEEP ONLY ALPHABETS
review=re.sub('[^a-zA-Z]',' ',dataset["Review"][0])
#not to remove character: ^a-zA-Z; to rplace the characters removed by space(' ');from string
#STEP 2: CONVERT TO LOWER CASE
review=review.lower()

#STEP 3: REMOVE UNECESARY WORDS SUCH AS THE,THIS,AND ETC.
import nltk
nltk.download("stopwords")#TO UPDATE THIS SET ONLY, NOT NECESSARY TO WRITE AGAIN AND AGAIN
from nltk.corpus import stopwords

###GO THROUGH WORDS ONE BY ONE IN REVIEW using stopwords from nltk.corpus
###review=review.split()
###review=[word for word in review if word not in stopwords.words("english") ]

#STEP 4: STEMMING + REMOVING UNNECESARY WORDS
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
review=review.split()
review=[ps.stem(word) for word in review if word not in stopwords.words("english") ]

#STEP 5: JOIN THE WORDS AGAIN
review=" ".join(review)
"""
#CLEANING EACH REVIEW
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
ps=PorterStemmer()
for i in dataset["Review"]:
    review=re.sub("[^a-zA-Z]"," ",i)
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if word not in stopwords.words('english')]
    review=" ".join(review)
    corpus.append(review)

#CREATE BAG OF WORDS MODEL USING TOKENIZATION:
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

"""COMMON MODELS: Naive BAyes; Random Forest, Decision Tree"""
"""1)NAIVE BAYES"""
# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier1 = GaussianNB()
classifier1.fit(x_train, y_train)

# Predicting the Test set results
y_pred1 = classifier1.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)

"""2) Decision Tree """
from sklearn.tree import DecisionTreeClassifier
classifier2=DecisionTreeClassifier(criterion="entropy",random_state=0)
classifier2.fit(x_train, y_train)
y_pred2 = classifier2.predict(x_test)
cm2 = confusion_matrix(y_test, y_pred2)

"""3)  RANDOM FOREST """
from sklearn.ensemble import RandomForestClassifier
classifier3=RandomForestClassifier(criterion="entropy",n_estimators=25,random_state=0)
classifier3.fit(x_train, y_train)
y_pred3 = classifier3.predict(x_test)
cm3 = confusion_matrix(y_test, y_pred3)

"""4)  Logistic Regression """
from sklearn.linear_model import LogisticRegression
classifier4=LogisticRegression(random_state=0)
classifier4.fit(x_train, y_train)
y_pred4 = classifier4.predict(x_test)
cm4 = confusion_matrix(y_test, y_pred4)

"""5+6)  SVM """
from sklearn.svm import SVC
classifier5=SVC(kernel='rbf',random_state=0)
classifier5.fit(x_train, y_train)
y_pred5 = classifier5.predict(x_test)
cm5 = confusion_matrix(y_test, y_pred5)

classifier6=SVC(kernel='poly',random_state=0,degree=2)
classifier6.fit(x_train, y_train)
y_pred6= classifier6.predict(x_test)
cm6 = confusion_matrix(y_test, y_pred6)


accuracy=[((cm1[0][0]+cm1[1][1])/200),((cm2[0][0]+cm2[1][1])/200),((cm3[0][0]+cm3[1][1])/200),((cm4[0][0]+cm4[1][1])/200),
          ((cm5[0][0]+cm5[1][1])/200),((cm6[0][0]+cm6[1][1])/200) ]
print(accuracy)



