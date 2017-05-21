# add triples

from helpers import *

version = 5
cells = pickle.load(open('data/matrix_X.p', 'rb'))	
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))

betas_abs = np.abs(betas)
beta_all = np.sum(betas_abs, axis=0)

top = 10
indexes = np.argsort(-beta_all)[:top]
cell_top = cells[:, indexes]
cells_new = np.zeros((cells.shape[0], (top-2)*(top-1)*top/6))

counter = 0
for i in range(top):
	for j in range(i+1, top):
		for r in range(j+1, top):
			cell_add = cell_top[:, i] * cell_top[:, j] * cell_top[:, r]
			cells_new[:, counter] = cell_add

			counter += 1

if counter != cells_new.shape[1]: sys.exit('not right dimenstion')

cell_without_zeros = pickle.load(open('data/very_new_cells-70.p', 'rb')) 
very_new_cells = np.zeros((988, cells_new.shape[1] + cell_without_zeros.shape[1]))
very_new_cells[:, :cells_new.shape[1]] = cells_new
very_new_cells[:, cells_new.shape[1]:] = cell_without_zeros

print cells_new.shape
print cells_new
print cell_without_zeros.shape

print very_new_cells
print very_new_cells.shape

print np.sum(very_new_cells) / np.prod(very_new_cells.shape)
print np.sum(cells_new) / np.prod(cells_new.shape)
print np.sum(cell_without_zeros) / np.prod(cell_without_zeros.shape)

pickle.dump(very_new_cells, open('data/very_new_cells-triples-%d.p' % top, 'wb'))
