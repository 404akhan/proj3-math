# generate data, lambda

from helpers import *

train, test = get_tr_cv()
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 
labels = get_labels_2()
data = pickle.load(open('data/clean_data16M.p', 'rb'))

version = 300
num = 265
betas = np.zeros((2, num, cells.shape[1]))
betas0 = np.zeros((2, num))

print 'len_data', len(data)
print 'beta size', betas.shape, betas0.shape
print np.sum(labels==0), np.sum(labels==1), np.sum(labels==labels)

lr = 0.01
aver_loss, aver_corr = None, None
bsize = 10000
arr_loss = []
lamb = 3*1e-5
print 'version', version, 'lamb', lamb

for epoch in range(5):
	print 'EPOCH %d STARTED' % (epoch+1)
	accum_grad = np.zeros_like(betas)
	accum_grad0 = np.zeros_like(betas0)
	accum_loss = 0
	accum_correct = 0
	for iteration, sample in enumerate(data):
		k, i, j, target = sample

		y = np.tanh( 
			np.dot(betas[labels[k], i, :], cells[k, :]) - np.dot(betas[labels[k], j, :], cells[k, :]) + \
				betas0[labels[k], i] - betas0[labels[k], j]
		)
		L = 1./2 * (y - target)**2
		accum_correct += (y>0 and target==1) or (y<=0 and target==-1)
		dy = y - target
		da = (1 - y**2) * dy
		dbi = cells[k, :] * da + lamb*np.sign(betas[labels[k], i, :])
		dbj = -cells[k, :] * da + lamb*np.sign(betas[labels[k], j, :])
		db0i = da
		db0j = -da

		accum_grad[labels[k], i, :] += dbi
		accum_grad[labels[k], j, :] += dbj
		accum_grad0[labels[k], i] += db0i
		accum_grad0[labels[k], j] += db0j
		accum_loss += L

		if (iteration+1) % bsize == 0:
			betas -= lr * accum_grad
			betas0 -= lr * accum_grad0

			accum_loss_div = accum_loss/bsize
			accum_correct /= 1.*bsize
			aver_loss = accum_loss_div if aver_loss is None else accum_loss_div * 0.01 + aver_loss * 0.99
			aver_corr = accum_correct if aver_corr is None else accum_correct * 0.01 + aver_corr * 0.99
			arr_loss.append(accum_loss_div)
			if (iteration+1) % (bsize*10) == 0: 
				print 'iter %d, cur_loss %.2f, aver_loss %.2f, accum_correct %.2f, aver_corr %.2f' % \
					(iteration+1, accum_loss_div, aver_loss, accum_correct, aver_corr)

			accum_grad = np.zeros_like(betas)
			accum_grad0 = np.zeros_like(betas0)
			accum_loss = 0
			accum_correct = 0
	lr /= 2

	correct = 0
	total = 0
	for iteration, sample in enumerate(test):
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

pickle.dump(betas, open('models/betas-v%d.p'%version, 'wb'))
pickle.dump(betas0, open('models/betas0-v%d.p'%version, 'wb'))