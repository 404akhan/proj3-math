# divide data 4:1

from helpers import *

train, test = get_tr_cv(4)

print np.sum(train[:, 3]) / len(train)
print np.sum(test[:, 3]) / len(test)