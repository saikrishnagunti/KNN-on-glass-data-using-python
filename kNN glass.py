# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 21:21:15 2021

@author: shivani
"""

import pandas as pd
import numpy as np

glass= pd.read_csv("C:\\Users\\shivani\\Desktop\\data science\\module 18 ML K-NN\\glass.csv")

glass["Type"].value_counts()

##Checking for the data distribution of the data
data = glass.describe()
data
## As, there is difference in the scale of the values, we normalise the data.

def norm_fumc(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)

norm = norm_fumc(glass.iloc[:,0:9])
glass1 = glass.iloc[:,9]

##Splitting the data into train and test data using stratified sampling

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(norm,glass1,test_size = 0.4,stratify = glass1)

##Checking the distribution of the labels which are taken
glass["Type"].value_counts()
y_train.value_counts()
y_test.value_counts()

##Building the model
from sklearn.neighbors import KNeighborsClassifier as KN

model = KN(n_neighbors = 5)
model.fit(x_train,y_train)

##Finding the accuracy of the model on training data
train_accuracy = np.mean(model.predict(x_train)==y_train) ##73.4375%
train_accuracy
##Accuracy on test data
test_accuracy = np.mean(model.predict(x_test)==y_test) ##61.62%
test_accuracy
##Changing the K value

model2 = KN(n_neighbors = 9)
model2.fit(x_train,y_train)

##Accuracy on training data
train_two = np.mean(model2.predict(x_train)==y_train) ##64.06%
train_two
##Accuracy on test data
test_two = np.mean(model2.predict(x_test)==y_test) ## 61.62%
test_two
# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
for i in range (4,30,1):
    model = KN(n_neighbors = i)
    model.fit(x_train,y_train)
    train_acc = np.mean(model.predict(x_train)==y_train)
    test_acc = np.mean(model.predict(x_test)==y_test)
    acc.append([train_acc, test_acc])

import matplotlib.pyplot as plt

##Training accuracy plot
plt.plot(np.arange(4,30,1),[i[0] for i in acc],'bo-')

##Test accuracy plot

plt.plot(np.arange(4,30,1),[i[1] for i in acc],'ro-')

plt.legend(["train","test"])

model3 = KN(n_neighbors = 6)
model3.fit(x_train,y_train)

pred_train = model3.predict(x_train)
cross_tab = pd.crosstab(y_train,pred_train)
cross_tab

train_accuracy = np.mean(pred_train == y_train)
train_accuracy
##67.18%

pred_test = model3.predict(x_test)
cross_tab_test = pd.crosstab(y_test,pred_test)
cross_tab_test

test_accuracy=np.mean(pred_test ==y_test)
test_accuracy
## 60.46%


