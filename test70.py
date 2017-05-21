# generate data

from helpers import *

train, test = get_tr_cv()
pairs_train = pickle.load(open('data/pairs_train-K50-complete.p', 'rb'))
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 


# generating data
arr = []
num = 265
for k in range(50):
	for i in range(num):
		for j in range(i+1, num):
			if pairs_train[k][i][j]==1:
				arr.append([k, i, j, 1])
			if pairs_train[k][j][i]==1:
				arr.append([k, i, j, -1])

data = np.array(arr)
shuf_arr = range(0, data.shape[0])
random.seed(1003)
random.shuffle(shuf_arr)
data = data[shuf_arr, :]
len_data = len(data)
print 'data size', len_data

betas = np.zeros((num, cells.shape[1]))
betas0 = np.zeros((num))

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
	for iteration, sample in enumerate(data):
		k, i, j, target = sample

		y = np.tanh( 
			np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
		)
		L = 1./2 * (y - target)**2
		accum_correct += (y>0 and target==1) or (y<=0 and target==-1)
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
		if k>= 50: continue
		y = np.tanh( 
			np.dot(betas[i, :], cells[k, :]) - np.dot(betas[j, :], cells[k, :]) + betas0[i] - betas0[j]
		)
		predict = 1 if y > 0 else -1
		if predict == target:
			correct += 1
		total += 1
	print 'heldout validation', correct * 1.0 / total, 'total', total