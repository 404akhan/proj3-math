from helpers import *

cells = pickle.load(open('data/cells.p', 'rb'))

print cells
print cells.shape
print 'mean filled %f' % np.mean(cells)

x = np.sum(cells, axis=0)

print x.shape
cells2 = cells[:, x > 10] 

print cells2.shape



sys.exit()
x = np.sum(cells, axis=0)
y = np.sum(cells, axis=1)
x.sort()
y.sort()

print 'x'
print x[:998]
print 'y'
print y

