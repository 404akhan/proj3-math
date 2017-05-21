# add one-hot feature classes

from helpers import * 

cells = get_matrix_X()
cells_add = np.zeros((cells.shape[0], 32))

index2name_cell = get_index2name_cell()
name2index_cancer = get_name2index_cancer()
df_local = pd.read_excel('data/cell_line_features.xlsx')

def map_cell_id2cancer_id_local(cell_id):
	name_cell = index2name_cell[cell_id]
	name_cancer = df_local[name_cell][0]
	return name2index_cancer[name_cancer]

for cell_id in range(cells_add.shape[0]):
	cancer_id = map_cell_id2cancer_id_local(cell_id)
	cells_add[cell_id, :] = np.eye(32, 32)[cancer_id]

print cells_add
print np.sum(cells_add)
print cells_add.shape

cells_new = np.zeros((cells.shape[0], cells.shape[1] + cells_add.shape[1]))
cells_new[:, :cells.shape[1]] = cells
cells_new[:, cells.shape[1]:] = cells_add

pickle.dump(cells_new, open('data/cells_new_cancer_one_hot.p', 'wb'))

