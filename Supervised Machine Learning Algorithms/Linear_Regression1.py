import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
#%% # Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True) #separates data n labels
print('Data dimensions ',np.shape(diabetes_X))
print('Label dimension',np.shape(diabetes_y))
print(diabetes_X[:2,:5])
#%% # Use only one feature
#using newaxis shape will be (4,1) otherwise (4,)
diabetes_X = diabetes_X[:, np.newaxis, 6] 
#diabetes_X2 = diabetes_X[:,2] #shape will be (4,)
#%% Split data
# Split the data into training/testing sets
X_train = diabetes_X[:-20] #exclude last 20 instances
X_test = diabetes_X[-20:]  #include last 20 instances

# Split the targets into training/testing sets
y_train = diabetes_y[:-20]
y_test = diabetes_y[-20:]

print('\nShape of training data',np.shape(X_train))
print('Shape of training Labels',np.shape(y_train))

print('\nShape of testing data',np.shape(X_test))
print('Shape of testing Labels',np.shape(y_test))
#%%
# Create linear regression object
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, y_train)
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
plt.scatter(X_test, y_test, color="red")
plt.plot(X_test, y_pred, color="blue", linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()











