# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:18:57 2023

@author: Zubair Kandhro
"""


# Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


# Importing the dataset
path = r"C:/Practical/data_banknote_authentication.csv"
df = pd.read_csv(path)

X = df.iloc[:, 0:4]
y = df.iloc[:, -1]

#print(df.shape)
#print(df.head())

# Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Training the kernel SVM model

classifier = SVC(kernel='linear',C=1.0,gamma="scale")
classifier.fit(X_train, y_train)

# Predicting the Values
y_pred = classifier.predict(X_test)

# Evaluation
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


#feature Scaling   z=(X-u/s)
#from sklearn.preprocessing import StandardScaler    
#st_x= StandardScaler()  
#x_train= st_x.fit_transform(x_train)    
#x_test= st_x.transform(x_test) 