# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:14:21 2023

@author: Zubair Kandhro
"""


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)


print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))
