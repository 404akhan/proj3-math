# project 3, math 6380J, Akhan Ismailov
# main piece of code, you will not be able to run it without preprocessed data
# you may find data at https://github.com/404akhan/proj3-math 

from helpers import * 

### create pairwise table i connect to j iff pairs_train[i][j]=1
total = 988
train = get_whole_data()
drug_num = get_total_drug()
pairs_train = np.zeros((total, drug_num, drug_num))

for sample in train:
	k, i, j, target = sample

	if target == 1: pairs_train[k][i][j] += 1
	if target == -1: pairs_train[k][j][i] += 1

####Floyd-Warshall algorithm, fill pairs_train[i][j] iff from i to j exist directed path
num = 265
for k in range(total):
	last = np.mean(pairs_train[k])
	print '\nstarting k', k, 'starting value', last
	for time in range(10):
		for i in range(num):
			for j in range(num):
				if pairs_train[k][i][j] == 1:
					pairs_train[k][i, :] = (pairs_train[k][i, :]+pairs_train[k][j, :] > 0).astype(int)
		new = np.mean(pairs_train[k])
		print time, new	
		
		if last == new:
			break
		else:
			last = new

### transform above pairs_train table, format (size, 4) data for training
print 'shape of data', pairs_train.shape
arr = []
num = 265
for k in range(pairs_train.shape[0]):
	for i in range(num):
		for j in range(i+1, num):
			if pairs_train[k][i][j]==1:
				arr.append([k, i, j, 1])
			if pairs_train[k][j][i]==1:
				arr.append([k, i, j, -1])

# shuffle data
data = np.array(arr)
shuf_arr = range(0, data.shape[0])
random.seed(1003)
random.shuffle(shuf_arr)
data = data[shuf_arr, :]
len_data = len(data)
print 'data size', len_data

### load cells 988 * 1595, (total_cells, num_features), including generated features
cells = pickle.load(open('data/cells_final_combo.p', 'rb')) 
### load labels for two clusters
labels = get_labels_2()
### betas and some parameters, 2 stands for num clusters
version = 1000
num = 265
betas = np.zeros((2, num, cells.shape[1]))
betas0 = np.zeros((2, num))

print 'len_data', len(data)
print 'beta size', betas.shape, betas0.shape
print np.sum(labels==0), np.sum(labels==1), np.sum(labels==labels)

### some variables for gradient descent
lr = 0.01
aver_loss, aver_corr = None, None
bsize = 10000
arr_loss = []
lamb = 3*1e-5
print 'version', version, 'lamb', lamb

### Gradient Descent starts here
for epoch in range(10):
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

	pickle.dump(betas, open('models/betas-v%d-%d.p'%(version, epoch), 'wb'))
	pickle.dump(betas0, open('models/betas0-v%d-%d.p'%(version, epoch), 'wb'))
