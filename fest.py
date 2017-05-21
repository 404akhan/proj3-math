from helpers import *

data = pickle.load(open('data/train2p6M_indexes.p', 'rb'))

print data

print np.sum(data[:, 3]==0), np.sum(data[:, 3]==data[:, 3])