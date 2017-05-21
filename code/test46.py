from helpers import *

pairs = pickle.load(open('data/pairs_tensor.p', 'rb'))
total_comp = pickle.load(open('data/total_comp_tensor.p', 'rb'))
total_comp[total_comp == 0] = 1

confidence = np.zeros_like(pairs)
for i in range(32):
	confidence[i, :, :] = pairs[i, :, :]*1. / total_comp[i, :, :]

print confidence
print 'stats accross axis 0, mean'
print np.mean(confidence, axis=0)
print 'std'
print np.std(confidence, axis=0)