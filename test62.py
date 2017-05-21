# fill values for cell_id a > b, b > c => a > c

from helpers import * 

train, test = get_tr_cv()
drug_num = get_total_drug()
pairs = np.zeros((50, drug_num, drug_num))
pairs_test = np.zeros((50, drug_num, drug_num))

for sample in train:
	k, i, j, target = sample

	if k < 50:
		if target == 1: pairs[k][i][j] += 1
		if target == -1: pairs[k][j][i] += 1

for sample in test:
	k, i, j, target = sample

	if k < 50:
		if target == 1: pairs_test[k][i][j] += 1
		if target == -1: pairs_test[k][j][i] += 1


pickle.dump(pairs, open('data/pairs_train-K0t50.p', 'wb'))
pickle.dump(pairs_test, open('data/pairs_test-K0t50.p', 'wb'))

print np.sum(pairs==0)+np.sum(pairs==1), np.prod(pairs.shape)
print np.sum(pairs_test==0)+np.sum(pairs_test==1), np.prod(pairs_test.shape)