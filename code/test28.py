from sklearn.cluster import KMeans
from helpers import *

X = get_matrix_X()

print X.shape
kmeans = KMeans(n_clusters=2, random_state=0).fit(X)

print kmeans.labels_
print kmeans.predict([np.zeros(X.shape[1]), np.ones(X.shape[1]), 
						np.random.rand(X.shape[1]), np.random.rand(X.shape[1]),
						np.random.rand(X.shape[1]), np.random.rand(X.shape[1])])
print kmeans.cluster_centers_
