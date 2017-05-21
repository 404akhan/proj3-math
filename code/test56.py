# choose best combination ~500

from helpers import *

cells = pickle.load(open('data/very_new_cells_one_hot-%d.p'%70, 'rb')) 
betas = pickle.load(open('models/betas-v%d.p'%66, 'rb'))

betas_abs = np.abs(betas)
beta_all = np.sum(betas_abs, axis=0)
indexes = np.argsort(-beta_all)

print cells.shape
print betas.shape

arr = indexes >= 1095
print arr[:100]

top500 = []
for iteration, index in enumerate(indexes):
	if index >= 1095:
		top500.append(index)
	if len(top500) == 500: 
		print 'at iteration %d out of %d breaking' % (iteration, len(indexes))
		break

top500_cells = cells[:, top500]
cells_first = pickle.load(open('data/cells_new_cancer_one_hot.p', 'rb')) 

cells_final_combo = np.zeros((988, cells_first.shape[1] + top500_cells.shape[1]))
cells_final_combo[:, :cells_first.shape[1]] = cells_first
cells_final_combo[:, cells_first.shape[1]:] = top500_cells

pickle.dump(cells_final_combo, open('data/cells_final_combo.p', 'wb'))

sys.exit()
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
