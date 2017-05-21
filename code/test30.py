# ensemble
# kmeans clustering

from helpers import *
import matplotlib.pyplot as plt

cells = get_matrix_X()
print 'shape of feauters (cluster)', cells.shape
labels = get_labels(cells)

train, test = get_tr_cv()
betas = pickle.load(open('models/betas-v%d.p'%50, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%50, 'rb'))

##############
cells2 = pickle.load(open('data/very_new_cells-70.p', 'rb')) 
betas2 = pickle.load(open('models/betas-v%d.p'%20, 'rb'))
betas02 = pickle.load(open('models/betas0-v%d.p'%20, 'rb'))
print 'shape of feauters (generated)', cells2.shape

correct = 0
total = 0
diff = []
for iteration, sample in enumerate(test):
	k, i, j, target = sample
	y = np.tanh( 
		np.dot(betas[labels[k], i, :], cells[k, :]) - \
		np.dot(betas[labels[k], j, :], cells[k, :]) + \
		betas0[labels[k], i] - betas0[labels[k], j]
	)
	y2 = np.tanh( 
		np.dot(betas2[i, :], cells2[k, :]) - np.dot(betas2[j, :], cells2[k, :]) + betas02[i] - betas02[j]
	)
	predict = 1 if y2 > 0 else -1
	if predict == target:
		correct += 1
	total += 1
	diff.append(np.abs(y-y2))

print np.mean(diff), np.std(diff)
print 'heldout validation', correct * 1.0 / total


