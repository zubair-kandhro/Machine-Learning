# -*- coding: utf-8 -*-
"""
Created on Thu May 25 16:38:28 2023

@author: Zubair Kandhro
"""

import numpy as np
import pandas as pd

df = pd.read_csv(r"C:\Users\Lenovo\Desktop\LAB AI\company.csv")
print(df.head())

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.preprocessing import LabelEncoder

le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()


X['company'] = le_company.fit_transform(X['company'])
X['job'] = le_company.fit_transform(X['job'])
X['degree'] = le_company.fit_transform(X['degree'])

print(X.head())

from sklearn import tree

model = tree.DecisionTreeClassifier()

model.fit(X, y)

print(X)

