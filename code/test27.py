from helpers import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

X = get_matrix_X() # np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
arr = model.fit_transform(X)
print arr.shape

kmeans = GaussianMixture(n_components=3, random_state=121).fit(X)
labels = kmeans.predict(X)
print labels

plt.plot(arr[(labels==2), 0], arr[(labels==2), 1], 'y.')
plt.plot(arr[(labels==1), 0], arr[(labels==1), 1], 'b.')
plt.plot(arr[(labels==0), 0], arr[(labels==0), 1], 'r.')
plt.show()

print np.sum(labels==0), np.sum(labels==1), np.sum(labels==2), np.sum(labels==3)

def get_labels(cells):
	kmeans = KMeans(n_clusters=2, random_state=0).fit(cells)
	return kmeans.labels_.astype(bool)

