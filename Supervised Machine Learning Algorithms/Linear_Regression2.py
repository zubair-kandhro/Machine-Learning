
import pandas as pd
import numpy as np
path = r"Boston_data.txt"
data_df = pd.read_csv(path, sep="\s+",header=None)
#path = r"D:\IBA-Semesters\Datasets\Bostonorg.txt"
#data_df1 = pd.read_csv(path, sep="\s+", skiprows=22, header=None)
data_x = data_df.values[1:,:-1]
data_y = data_df.values[1:,np.newaxis,-1]
#%% Split data
# Split the data into training/testing sets
SplitRatio = 0.3
Ts_Instances = round(len(data_x)*SplitRatio)
X_train = data_x[:-Ts_Instances] 
X_test = data_x[-Ts_Instances:]  
# Split the targets into training/testing sets
y_train = data_y[:-Ts_Instances]
y_test = data_y[-Ts_Instances:]

print('\nShape of training data',np.shape(X_train))  #diabetes_X[:, np.newaxis, 6] 
print('Shape of training Labels',np.shape(y_train))

print('\nShape of testing data',np.shape(X_test))
print('Shape of testing Labels',np.shape(y_test))

#Use one feature
Fet = 12
X_train = X_train[:,np.newaxis,Fet] 
X_test = X_test[:,np.newaxis,Fet]  
#X_train = X_train[:,Fet] 
#X_test = X_test[:,Fet]  

#%%
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Create and train linear regression object
#Col = 0
regr = linear_model.LinearRegression().fit(X_train, y_train)

#%%
# Make predictions using the testing set
y_pred = regr.predict(X_test)
#%%
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error (MSE): %.2f" % mean_squared_error(y_test, y_pred))
# The coefficient of determination: 1 is perfect prediction
# Coefficient of determination also called as R2 score 
# is used to evaluate the performance of a linear regression model.
# It is the amount of the variation in the output dependent attribute 
# which is predictable from the input independent variable(s)
print("Coefficient of determination: %.2f" % r2_score(y_test, y_pred))#%%
#%% Plot outputs
plt.scatter(X_test.astype(np.float64), y_test.astype(np.float64), color="red")
plt.plot(X_test.astype(np.float64), y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()


