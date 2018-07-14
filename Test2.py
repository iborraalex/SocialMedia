# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 19:26:59 2018

@author: ALEJANDRO IBORRA
"""

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import time
import random
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd


start_time = time.time()

random.seed(1234)


dataset = pd.read_csv("lista_modelo.txt")



# Importing the dataset
#X = dataset.iloc[:, 1:2007].values
#y = dataset.iloc[:, 2007].values

X = dataset.iloc[:, 1:2007].values
y = dataset.iloc[:, 2007].values


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)



 
start_time_svm = time.time()
##  SVM  ##

model_svm = svm.SVC(kernel = 'linear', C=1, gamma = 1)
model_svm.fit(X_train,y_train)
predicted_svm = model_svm.predict(X_test)
accuracy_svm = accuracy_score(y_test,predicted_svm)
print("accuracy SVM: ",accuracy_svm," time: ",time.time()-start_time_svm)


start_time_dt = time.time()
##  Decision Tree  ##

model_dt = DecisionTreeClassifier()
model_dt.fit(X_train,y_train)
predicted_dt = model_dt.predict(X_test)
accuracy_dt = accuracy_score(y_test,predicted_dt)
print("accuracy Decision Tree: ",accuracy_dt," time: ",time.time()-start_time_dt)


start_time_rf = time.time()
##  Random Forest  ##

model_rf = RandomForestClassifier(n_jobs=2, random_state=0)
model_rf.fit(X_train,y_train)
predicted_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test,predicted_rf)
print("accuracy Random Forest:",accuracy_rf," time: ",time.time()-start_time_rf)



start_time_nn = time.time()
##  Neural Net  ##

model_nn = MLPClassifier(hidden_layer_sizes=(22,22,22),max_iter=500)
model_nn.fit(X_train,y_train)
predicted_nn = model_nn.predict(X_test)
accuracy_nn = accuracy_score(y_test,predicted_nn)
print("accuracy Neural Net: ", accuracy_nn," time: ",time.time()-start_time_nn)



print('It took {0:0.2f} seconds'.format(time.time() - start_time))