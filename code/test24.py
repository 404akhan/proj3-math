# logistic

from helpers import *
import matplotlib.pyplot as plt

train, test = get_tr_cv()
cells = pickle.load(open('data/very_new_cells-70.p', 'rb')) 
betas = np.zeros((get_total_drug(), cells.shape[1]))
betas0 = np.zeros((get_total_drug()))
print train.shape
print cells.shape
print betas.shape
print betas0.shape

version = 40
lr = 0.01
aver_loss, aver_corr = None, None
bsize = 10000
arr_loss = []

for epoch in range(10):
	print 'EPOCH %d STARTED' % (epoch+1)
	accum_grad = np.zeros_like(betas)
	accum_grad0 = np.zeros_like(betas0)
	accum_loss = 0
	accum_correct = 0
	for iteration, sample in enumerate(train):
		k, i, j, target = sample
		if target == -1: target = 0

		y = sigmoid( 
			np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
		)
		L = (y - target)**2
		accum_correct += (y>0.5 and target==1) or (y<=0.5 and target==0)
		da = y - target
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

	correct_thres = 0
	correct_sample = 0
	total = 0
	for iteration, sample in enumerate(test):
		k, i, j, target = sample
		y = sigmoid( 
			np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
		)
		predict_thres = 1 if y > 0.5 else -1
		predict_sample = 1 if np.random.rand() < y else -1
		if predict_thres == target:	correct_thres += 1
		if predict_sample == target: correct_sample += 1
		total += 1
	print 'heldout validation, thres', correct_thres * 1.0 / total
	print 'heldout validation, sample', correct_sample * 1.0 / total

pickle.dump(betas, open('models/betas-v%d.p'%version, 'wb'))
pickle.dump(betas0, open('models/betas0-v%d.p'%version, 'wb'))

plt.plot(arr_loss)
plt.show()