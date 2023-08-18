# NB with multiclass
#Load dataset
from sklearn import datasets
wine = datasets.load_wine()
#%% Print Features and Label
print ("Features: ", wine.feature_names)
print ("Labels: ", wine.target_names)

#%%
#Features:  ['alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'magnesium', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols', 
#           'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines', 'proline']
#Labels:  ['class_0' 'class_1' 'class_2']

#%% print data(feature)shape
print(wine.data.shape)

#%% print the wine data features (top 5 records)
print (wine.data[0:5])

#%% print the wine labels (0:Class_0, 1:class_2, 2:class_2)
print (wine.target)

#%% data split (#conda install -c "conda-forge/label/scikit-learn_rc" scikit-learn)
from sklearn.model_selection import train_test_split

X=wine.data
y=wine.target
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109) # 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True) # 70% training and 30% test

#%% Model training and prediction
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = gnb.predict(X_test)

#%% Evaluation
from sklearn import metrics
print("\n conFusion Matrix \n",metrics.confusion_matrix(y_test,y_pred))
print("\nAccuracy:",metrics.accuracy_score(y_test, y_pred))

