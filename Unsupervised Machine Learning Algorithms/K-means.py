from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#%% synthatic data
features, true_labels = make_blobs( n_samples=200, centers=3, cluster_std=2.75, random_state=42)
print(features[:5])
print(true_labels[:5])
#%% normalization
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
print(scaled_features[:5])

#%% use kmenas
kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42)
kmeans.fit(scaled_features)
#%% Evaluation
#Lowest sum of the squared error (SSE) / inertia
print("Sum of the squared error (SSE): ",kmeans.inertia_)
# Final locations of the centroid
print("Cluster locations: \n",kmeans.cluster_centers_)
# The number of iterations required to converge
print("Number of Iterations: ",kmeans.n_iter_)
# Labels
print("Labels: ",kmeans.labels_[:5])
print("Clsuter Labels: ", set(kmeans.labels_))



