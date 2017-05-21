# submit and test

from helpers import *

train, test = get_tr_cv()
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 
labels = get_labels_2()

version = 600
num = 265
betas = pickle.load(open('models/betas-v%d-%d.p'%(version,6), 'rb'))
betas0 = pickle.load(open('models/betas0-v%d-%d.p'%(version,6), 'rb'))
print 'beta size', betas.shape, betas0.shape
print np.sum(labels==0), np.sum(labels==1), np.sum(labels==labels)
print 'version', version

correct = 0
total = 0
for iteration, sample in enumerate(train):
	k, i, j, target = sample
	y = np.tanh( 
		np.dot(betas[labels[k], i, :], cells[k, :]) - np.dot(betas[labels[k], j, :], cells[k, :]) + \
			betas0[labels[k], i] - betas0[labels[k], j]
	)
	predict = 1 if y > 0 else -1
	if predict == target:
		correct += 1
	total += 1
print 'heldout validation', correct * 1.0 / total, 'total', total
