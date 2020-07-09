# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:36:22 2020

@author: Aditi Jain
"""

import pandas as pd

dataset = pd.read_csv('bank_data.csv')

X = dataset.iloc[:, [0,2,4,5]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'poly', random_state = 0)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)



