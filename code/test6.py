# code to test test5.py

from helpers import *
import matplotlib.pyplot as plt

train, test = get_tr_cv()
print train.shape
cells = get_matrix_X()
print cells.shape
betas = np.zeros((get_total_drug(), get_total_features()))
print betas.shape

betas = pickle.load(open('models/betas-v1.p', 'rb'))

correct = 0
total = 0
for iteration, sample in enumerate(test):
	k, i, j, target = sample

	y = np.tanh( 
		np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :])
	)

	predict = 1 if y > 0 else -1
	if predict == target:
		correct += 1
	total += 1

print correct * 1.0 / total
