from helpers import *

cells = pickle.load(open('data/matrix_X.p', 'rb'))	

version = 5
betas = pickle.load(open('models/betas-v%d.p'%version, 'rb'))

print cells.shape
print betas.shape

print cells
print betas

betas_abs = np.abs(betas)
beta_all = np.sum(betas_abs, axis=0)

print beta_all.shape
print beta_all[:600]
print beta_all[600:]

top = 80
indexes = np.argsort(-beta_all)[:top]

print indexes

print cells.shape

cell_top = cells[:, indexes]

cells_new = np.zeros((cells.shape[0], top/2*top))

counter = 0
for i in range(top):
	for j in range(i+1, top):
		cell_add = cell_top[:, i] * cell_top[:, j]
		cells_new[:, counter] = cell_add

		counter += 1

print cells_new.shape
print cells_new

cell_without_zeros = get_matrix_X()
print cell_without_zeros.shape

very_new_cells = np.zeros((988, cells_new.shape[1] + cell_without_zeros.shape[1]))

very_new_cells[:, :cells_new.shape[1]] = cells_new
very_new_cells[:, cells_new.shape[1]:] = cell_without_zeros

print very_new_cells
print very_new_cells.shape

print np.sum(very_new_cells) / np.prod(very_new_cells.shape)
print np.sum(cells_new) / np.prod(cells_new.shape)
print np.sum(cell_without_zeros) / np.prod(cell_without_zeros.shape)

pickle.dump(very_new_cells, open('data/very_new_cells-%d.p' % top, 'wb'))
