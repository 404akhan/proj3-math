# create classes, assign each cell line id to a class, as for 1250 average cell_line_id that belongs
# to same class

from helpers import *

def get_keys_cells():
	name2index_cell = get_name2index_cell()
	return name2index_cell.keys()

def get_name2index_cancer():
	df = pd.read_excel('data/cell_line_features.xlsx')
	arr = np.array(df)

	cancer_types = {}
	for c_t in arr[0]:
		cancer_types[c_t] = 0

	counter = 0
	for k in cancer_types:
		cancer_types[k] = counter
		counter += 1

	return cancer_types

def get_index2name_cancer():
	index2name_cancer = get_name2index_cancer()
	return {v: k for k, v in index2name_cancer.iteritems()}

def get_cancer_name(cell_name):
	df = pd.read_excel('data/cell_line_features.xlsx')
	return df[cell_name][0]

def map_cell_id2cancer_id(index2name_cell, name2index_cancer, cell_id):
	name_cell = index2name_cell[cell_id]
	name_cancer = get_cancer_name(name_cell)
	print name_cell, name_cancer
	return name2index_cancer[name_cancer]

index2name_cell = get_index2name_cell()
name2index_cancer = get_name2index_cancer()

cancer_id = map_cell_id2cancer_id(index2name_cell, name2index_cancer, 0)
print cancer_id

