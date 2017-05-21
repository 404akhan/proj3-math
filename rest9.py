# clean gradient
# cluster + 1500 feature

from helpers import *
import matplotlib.pyplot as plt

train, test = get_tr_cv()
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 
labels = np.array(get_labels_2())
betas = np.zeros((2, get_total_drug(), cells.shape[1]))
betas0 = np.zeros((2, get_total_drug()))
print train.shape
print cells.shape
print betas.shape
print betas0.shape
print np.sum(labels==0), np.sum(labels==1), np.sum(labels==labels)

version = 1
lr = 0.01
aver_loss, aver_corr = None, None
bsize = 10000
arr_loss = []
lamb = 3*1e-5
len_train = len(train)
print 'lambda', lamb, 'version', version

for epoch in range(10):
	print 'EPOCH %d STARTED' % (epoch+1)
	accum_loss = 0
	accum_correct = 0
	for iteration in range(len_train/bsize-1):
		k = train[iteration*bsize:(iteration+1)*bsize, 0]
		i = train[iteration*bsize:(iteration+1)*bsize, 1]
		j = train[iteration*bsize:(iteration+1)*bsize, 2]
		target = train[iteration*bsize:(iteration+1)*bsize, 3]

		print i
		print betas[np.zeros(len(k), type=np.int), np.zeros(len(k), type=np.int)]
		print betas[np.zeros(len(k)), i].shape
		print betas[labels[k], i].shape
		y = np.tanh( 
			np.sum(betas[labels[k], i, :]*cells[k, :], axis=1) - \
			np.sum(betas[labels[k], j, :]*cells[k, :], axis=1) + \
			betas0[labels[k]][i] - betas0[labels[k]][j]
		)
		L = 1./2 * (y - target)**2
		accum_correct += (y>0 and target==1) or (y<=0 and target==-1)
		dy = y - target
		da = (1 - y**2) * dy
		dbi = cells[k, :] * da + lamb*np.sign(betas[labels[k], i, :])
		dbj = -cells[k, :] * da + lamb*np.sign(betas[labels[k], j, :])
		db0i = da
		db0j = -da

		print dbi.shape
		print dbj.shape
		sys.exit()
		betas[labels[k], i, :] += dbi
		betas[labels[k], j, :] += dbj
		betas0[labels[k], i] += db0i
		betas0[labels[k], j] += db0j
		accum_loss += L

		if (iteration+1) % bsize == 0:
			betas -= lr * accum_grad
			betas0 -= lr * accum_grad0

			accum_loss_div = accum_loss/bsize
			accum_correct /= 1.*bsize
			aver_loss = accum_loss_div if aver_loss is None else accum_loss_div * 0.01 + aver_loss * 0.99
			aver_corr = accum_correct if aver_corr is None else accum_correct * 0.01 + aver_corr * 0.99
			arr_loss.append(accum_loss_div)
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
	print 'heldout validation', correct * 1.0 / total

pickle.dump(betas, open('models/betas-v%d.p'%version, 'wb'))
pickle.dump(betas0, open('models/betas0-v%d.p'%version, 'wb'))

plt.plot(arr_loss)
plt.show()