# add top 30 to one hot and top 70, if performance not becoming super worse do two clusters

from helpers import *

cells = get_matrix_X() #pickle.load(open('data/cells_new_cancer_one_hot.p', 'rb')) 
betas = pickle.load(open('models/betas-v%d.p'%65, 'rb'))

betas_abs = np.abs(betas)
beta_all = np.sum(betas_abs, axis=0)
top = 70
indexes = np.argsort(-beta_all)[:top]

cell_top = cells[:, indexes]
cells_new = np.zeros((cells.shape[0], top*(top-1)/2))

counter = 0
for i in range(top):
	for j in range(i+1, top):
		cell_add = cell_top[:, i] * cell_top[:, j]
		cells_new[:, counter] = cell_add
		counter += 1

if 988 != cells.shape[0]: sys.exit('eblan suka bug')
cells = pickle.load(open('data/cells_new_cancer_one_hot.p', 'rb')) 
very_new_cells = np.zeros((988, cells.shape[1] + cells_new.shape[1]))

very_new_cells[:, :cells.shape[1]] = cells
very_new_cells[:, cells.shape[1]:] = cells_new

print np.sum(very_new_cells) / np.prod(very_new_cells.shape)
print np.sum(cells) / np.prod(cells.shape)
print np.sum(cells_new) / np.prod(cells_new.shape)
print cells_new.shape

pickle.dump(very_new_cells, open('data/very_new_cells_one_hot-%d.p' % top, 'wb'))
