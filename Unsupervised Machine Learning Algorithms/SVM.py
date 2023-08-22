#Banknotes authetication datset
# Data were extracted from images that were taken for the evaluation of an authentication procedure for banknotes.
#https://archive.ics.uci.edu/ml/datasets/banknote+authentication

import pandas as pd
path = r"D:\IBA-Semesters\00-Previous Semesters Data\13-Spring2023\Datasets\bill_authentication.csv"
bankdata = pd.read_csv(path)

#%%
print(bankdata.shape)
print(bankdata.head())

#%% Preprocessing
X = bankdata.drop('Class', axis=1)
y = bankdata['Class']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

#%% Train model
# kernel = linear, sigmoid,rbf,poly(with degree)
from sklearn.svm import SVC
svclassifier = SVC(kernel='linear',C=1.0,gamma="scale")
svclassifier.fit(X_train, y_train)
y_pred = svclassifier.predict(X_test)

#%% Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

#%%
#feature Scaling   z=(X-u/s)
#from sklearn.preprocessing import StandardScaler    
#st_x= StandardScaler()  
#x_train= st_x.fit_transform(x_train)    
#x_test= st_x.transform(x_test) 