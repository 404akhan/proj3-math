# working logistic without b0

from helpers import *
import matplotlib.pyplot as plt

train, test = get_tr_cv()
print train.shape
cells = get_matrix_X()
print cells.shape
betas = np.zeros((get_total_drug(), get_total_features()))
print betas.shape

version = 3
lr = 0.01
aver_loss = None
bsize = 1000
arr_loss = []

for epoch in range(2):
	print 'EPOCH %d STARTED' % (epoch+1)
	accum_grad = np.zeros_like(betas)
	accum_loss = 0
	for iteration, sample in enumerate(train):
		k, i, j, target = sample

		y = np.tanh( 
			np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :])
		)
		L = 1./2 * (y - target)**2
		dy = y - target
		da = (1 - y**2) * dy
		dbi = cells[k, :] * da
		dbj = -cells[k, :] * da

		accum_grad[i, :] += lr * dbi
		accum_grad[j, :] += lr * dbj
		accum_loss += L

		if (iteration+1) % bsize == 0:
			betas -= lr * accum_grad

			aver_loss = accum_loss/bsize if aver_loss is None else accum_loss/bsize * 0.01 + aver_loss * 0.99
			arr_loss.append(accum_loss/bsize)
			print 'iter %d, cur_loss %.2f, aver_loss %.2f' % (iteration+1, accum_loss/bsize, aver_loss)

			accum_grad = np.zeros_like(betas)
			accum_loss = 0
	lr /= 3

pickle.dump(betas, open('data/betas-v%d.p'%version, 'wb'))

plt.plot(arr_loss)
plt.show()