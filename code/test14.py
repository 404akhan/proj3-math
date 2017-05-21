# adding b0

from helpers import *
import matplotlib.pyplot as plt

np.random.seed(7)

train, test = get_tr_cv()
cells = get_matrix_X()

hl_size = 200
W = np.random.randn(cells.shape[1], hl_size) / cells.shape[1]
betas = np.random.randn(get_total_drug(), hl_size) / hl_size
betas0 = np.zeros((get_total_drug()))

print train.shape
print cells.shape
print betas.shape
print betas0.shape

version = 12
lr = 0.01
aver_loss = None
bsize = 1000
arr_loss = []

for epoch in range(1):
	print 'EPOCH %d STARTED' % (epoch+1)
	accum_grad = np.zeros_like(betas)
	accum_grad0 = np.zeros_like(betas0)
	accum_gradW = np.zeros_like(W)
	accum_loss = 0
	for iteration, sample in enumerate(train):
		k, i, j, target = sample

		hl = np.matmul(cells[k, :], W)
		hl = sigmoid(hl)

		y = np.tanh( 
			np.dot(betas[i, :], hl) - np.dot(betas[j, :], hl) + betas0[i] - betas0[j]
		)
		L = 1./2 * (y - target)**2
		dy = y - target
		da = (1 - y**2) * dy
		dbi = hl * da
		dbj = -hl * da
		db0i = da
		db0j = -da
		dhl = (betas[i, :] - betas[j, :]) * da
		dhl *= hl * (1 - hl)
		dW = np.matmul(cells[k, :].reshape(W.shape[0], 1), dhl.reshape(1, W.shape[1]))

		accum_grad[i, :] += dbi
		accum_grad[j, :] += dbj
		accum_grad0[i] += db0i
		accum_grad0[j] += db0j
		accum_gradW += dW
		accum_loss += L

		if (iteration+1) % bsize == 0:
			print np.std(accum_grad), np.std(accum_grad0), np.mean(accum_gradW)
			betas -= lr * accum_grad
			betas0 -= lr * accum_grad0
			W -= lr * accum_gradW

			accum_loss_div = accum_loss/bsize
			aver_loss = accum_loss_div if aver_loss is None else accum_loss_div * 0.01 + aver_loss * 0.99
			arr_loss.append(accum_loss_div)
			print 'iter %d, cur_loss %.2f, aver_loss %.2f' % (iteration+1, accum_loss_div, aver_loss)

			accum_grad = np.zeros_like(betas)
			accum_grad0 = np.zeros_like(betas0)
			accum_gradW = np.zeros_like(W)
			accum_loss = 0
		
		if (iteration+1) % 100000 == 0:
			print 'hi'
			lr /= 1.5
		if iteration > 1000*1002: break

pickle.dump(betas, open('models/betas-v%d.p'%version, 'wb'))
pickle.dump(betas0, open('models/betas0-v%d.p'%version, 'wb'))
pickle.dump(W, open('models/W-v%d.p'%version, 'wb'))

plt.plot(arr_loss)
plt.show()