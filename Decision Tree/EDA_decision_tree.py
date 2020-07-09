# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 00:41:10 2020

@author: Aditi Jain
"""

import pandas as pd

dataset = pd.read_excel('alarm_files.xlsx')

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

#importing and fitting decision tree to Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#predicting test results
y_pred = classifier.predict(X_test)

#creating fusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
