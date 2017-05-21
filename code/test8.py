# code to test test7.py

from helpers import *
import matplotlib.pyplot as plt

train, test = get_tr_cv()
print train.shape, test.shape
cells = get_matrix_X()
print cells.shape

version = 5
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))
betas0 = pickle.load(open('models/betas0-v%d.p'%version, 'rb'))

correct = 0
total = 0
for iteration, sample in enumerate(test):
	k, i, j, target = sample

	y = np.tanh( 
		np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
	)

	predict = 1 if y > 0 else -1
	if predict == target:
		correct += 1
	total += 1

print correct * 1.0 / total
