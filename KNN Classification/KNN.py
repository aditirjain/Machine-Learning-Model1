# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:45:10 2020

@author: Aditi Jain
"""



import pandas as pd

dataset = pd.read_csv('bank_data.csv')

X = dataset.iloc[:,[0,2,4,5]].values
y = dataset.iloc[:, -1].values

#importing train_test_split from sklearn.model_selection to split data into training & testing sets
from sklearn.model_selection import train_test_split
X_train, X_test,y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#importing StandardScaler from sklearn.preprocessing to scale matrix of features
from sklearn.preprocessing import StandardScaler
scaled_X = StandardScaler()
X_train = scaled_X.fit_transform(X_train)
X_test = scaled_X.transform(X_test)


#importing KNeighborsClassifier from sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier

#fitting K-MN to training set
classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#Predicting y values using predict method in the class
y_pred = classifier.predict(X_test)

#creating confusion metrics
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)