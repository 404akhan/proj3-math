# adding b0

from helpers import *
import matplotlib.pyplot as plt

train = pickle.load(open('data/train04.p', 'rb'))
test = pickle.load(open('data/test45.p', 'rb'))
cells = pickle.load(open('data/cells.p', 'rb'))
W1 = pickle.load(open('models/ae-W1-v1.p', 'rb'))
hl = np.matmul(cells, W1)
cells = sigmoid(hl)

betas = np.zeros((get_total_drug(), cells.shape[1]))
betas0 = np.zeros((get_total_drug()))

print train.shape
print cells.shape
print betas.shape
print betas0.shape

version = 6
lr = 0.01
aver_loss = None
bsize = 10000
arr_loss = []

for epoch in range(3):
	print 'EPOCH %d STARTED' % (epoch+1)
	accum_grad = np.zeros_like(betas)
	accum_grad0 = np.zeros_like(betas0)
	accum_loss = 0
	for iteration, sample in enumerate(train):
		k, i, j, target = sample

		y = np.tanh( 
			np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
		)
		L = 1./2 * (y - target)**2
		dy = y - target
		da = (1 - y**2) * dy
		dbi = cells[k, :] * da
		dbj = -cells[k, :] * da
		db0i = da
		db0j = -da

		accum_grad[i, :] += dbi
		accum_grad[j, :] += dbj
		accum_grad0[i] += db0i
		accum_grad0[j] += db0j
		accum_loss += L

		if (iteration+1) % bsize == 0:
			betas -= lr * accum_grad
			betas0 -= lr * accum_grad0

			accum_loss_div = accum_loss/bsize
			aver_loss = accum_loss_div if aver_loss is None else accum_loss_div * 0.01 + aver_loss * 0.99
			arr_loss.append(accum_loss_div)
			print 'iter %d, cur_loss %.2f, aver_loss %.2f' % (iteration+1, accum_loss_div, aver_loss)

			accum_grad = np.zeros_like(betas)
			accum_grad0 = np.zeros_like(betas0)
			accum_loss = 0
	lr /= 3

pickle.dump(betas, open('models/betas-v%d.p'%version, 'wb'))
pickle.dump(betas0, open('models/betas0-v%d.p'%version, 'wb'))

plt.plot(arr_loss)
plt.show()