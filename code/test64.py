# combination

# cluster + 1500 feature

from helpers import *
import matplotlib.pyplot as plt

version = 120
train, test = get_tr_cv()
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 
labels = get_labels_2()
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%version, 'rb'))
print train.shape
print cells.shape
print betas.shape
print betas0.shape
print np.sum(labels==0), np.sum(labels==1), np.sum(labels==labels)

# method 2
print 'loading pairs_train'
pairs_train = pickle.load(open('data/pairs_train-Kall-complete.p', 'rb'))
correct_method2 = 0
total_method2 = 0
print 'finish loading pairs_train'
# end method 2

correct = 0
total = 0
for iteration, sample in enumerate(test):
	k, i, j, target = sample
	y = np.tanh( 
		np.dot(betas[labels[k], i, :], cells[k, :]) - np.dot(betas[labels[k], j, :], cells[k, :]) + \
			betas0[labels[k], i] - betas0[labels[k], j]
	)
	predict = 1 if y > 0 else -1

	# method 2
	if pairs_train[k][i][j] == 1: 
		predict = 1
		total_method2 += 1
		if predict == target: correct_method2 += 1
	if pairs_train[k][j][i] == 1: 
		predict = -1
		total_method2 += 1
		if predict == target: correct_method2 += 1
	# end method 2

	if predict == target: 
		correct += 1
	total += 1

# method 2	
print 'results from method 2', correct_method2, total_method2, correct_method2*1./total_method2
# end method 2
print 'heldout validation', correct * 1.0 / total

