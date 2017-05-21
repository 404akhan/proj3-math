# feature extraction, autoencoder

from helpers import *

train = pickle.load(open('data/train04.p', 'rb'))
cells = pickle.load(open('data/cells.p', 'rb'))

print train.shape
print cells.shape

print train
print cells

aver_mean_error, aver_std_error = None, None
lr = 0.01
version = 3

hl_size = 100
W1 = np.random.randn(1250, hl_size) / np.sqrt(1250)
W2 = np.random.randn(hl_size, 1250) / np.sqrt(hl_size)

for iteration in xrange(3*1000):
	hl = np.matmul(cells, W1)
	hl = sigmoid(hl)
	out = np.matmul(hl, W2)
	out = sigmoid(out)
	target = copy.deepcopy(cells)

	diff = np.abs(out - target)
	mean_error = np.mean(diff)
	std_error = np.std(diff)

	dout = out - target
	dout *= out * (1 - out)
	dhl = np.matmul(dout, W2.transpose())
	dhl *= hl * (1 - hl)

	dW2 = np.matmul(hl.transpose(), dout)
	dW1 = np.matmul(cells.transpose(), dhl)

	print np.mean(dW2), np.mean(dW1)
	W2 -= lr * dW2
	W1 -= lr * dW1

	aver_mean_error = get_aver_weight(aver_mean_error, mean_error)
	aver_std_error = get_aver_weight(aver_std_error, std_error)
	print 'iter %d, mean_error %.3f, aver_mean_error %.3f, std_error %.3f, aver_std_error %.3f' % \
		(iteration, mean_error, aver_mean_error, std_error, aver_std_error)

	if iter == 1000:
		lr = 0.003
	if iter == 2000:
		lr = 0.0001

pickle.dump(W1, open('models/ae-W1-v%d.p'%version, 'wb'))
pickle.dump(W2, open('models/ae-W2-v%d.p'%version, 'wb'))

