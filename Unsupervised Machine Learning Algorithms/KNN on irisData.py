# Import necessary modules
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
#%%  
irisData = load_iris()
  
# Create feature and target arrays
X = irisData.data
y = irisData.target
  
# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
             X, y, test_size = 0.3, random_state=42)
  
#%% Declare array to store accuracies of 10 k vales
n = 10
tr_acc = np.empty(n)
ts_acc = np.empty(n)

#%% Loop over K values
for i in range(0, 9):
    knn = KNeighborsClassifier(n_neighbors=i+1)
    knn.fit(X_train, y_train)
      
    # Compute training and test data accuracy
    tr_acc[i] = knn.score(X_train, y_train)
    ts_acc[i] = knn.score(X_test, y_test)
  
# Generate plot
plt.plot(, tr_acc, label = 'Training dataset Accuracy')
plt.plot(, ts_acc, label = 'Testing dataset Accuracy')

  
plt.legend()
plt.xlabel('k=n_neighbors')
plt.ylabel('Accuracy')
plt.show()