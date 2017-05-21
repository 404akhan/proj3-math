import numpy as np
import pandas as pd
import sys
import cPickle as pickle
import random
import copy
from sklearn.cluster import KMeans

def get_name2index_drug():
	df = pd.read_csv('data/train3M_drug_pair.csv')

	drug1 = np.array(df['Drug1'])
	drug2 = np.array(df['Drug2'])

	name2index = {} 

	for dr in drug1:
		name2index[dr] = 0

	for dr in drug2:
		name2index[dr] = 0

	counter = 0
	for i in name2index:
		name2index[i] = counter
		counter += 1

	return name2index

def get_index2name_drug():
	name2index = get_name2index_drug()

	return { v: k for k, v in name2index.iteritems() }

def get_total_drug():
	name2index = get_name2index_drug()

	return len([k for k in name2index])

def get_name2index_cell():
	df = pd.read_csv('data/train3M_drug_pair.csv')

	CellLineID = np.array(df['CellLineID'])

	name2index = {}

	for dr in CellLineID:
		name2index[dr] = 0

	counter = 0
	for i in name2index:
		name2index[i] = counter
		counter += 1

	return name2index

def get_index2name_cell():
	name2index = get_name2index_cell()

	return { v: k for k, v in name2index.iteritems() }

def get_total_cell():
	name2index = get_name2index_cell()

	return len([k for k in name2index])

def get_total_features():
	return 1250

def create_matrix_X():
	sys.exit('run protected')

	df = pd.read_excel('data/cell_line_features.xlsx')
	index2name_cell = get_index2name_cell()

	X = np.zeros((get_total_cell(), 1250))

	for i in range(X.shape[0]):
		X[i, :] = df[ index2name_cell[i] ][1:]

	pickle.dump(X, open('data/matrix_X.p', 'wb'))

def create_train_indexes():
	sys.exit('run protected')

	df = pd.read_csv('data/train3M_drug_pair.csv')
	data = np.array(df.query('Comparison == Comparison'))		

	name2index_drug = get_name2index_drug()
	name2index_cell = get_name2index_cell()

	data_numbers = np.zeros_like(data)
	for i in range(data.shape[0]):
		data_numbers[i][0] = name2index_cell[data[i][0]]
		data_numbers[i][1] = name2index_drug[data[i][1]]
		data_numbers[i][2] = name2index_drug[data[i][2]]
		data_numbers[i][3] = data[i][3]

	pickle.dump(data_numbers, open('data/train2p6M_indexes.p', 'wb'))

def get_whole_data():
	random.seed(1003)	
	data = pickle.load(open('data/train2p6M_indexes.p', 'rb'))
	shuf_arr = range(0, data.shape[0])
	random.shuffle(shuf_arr)
	data = data[shuf_arr, :]

	return data

def get_tr_cv(index_cv=4, seed=1003):
	random.seed(seed)

	data = pickle.load(open('data/train2p6M_indexes.p', 'rb'))

	shuf_arr = range(0, data.shape[0])
	random.shuffle(shuf_arr)
	data = data[shuf_arr, :]

	num_parts = 5
	size_part = data.shape[0] / num_parts
	arr_pieces = []
	for i in range(5):
		arr_pieces.append( data[i*size_part:(i+1)*size_part] )

	indexes = [(index_cv + add) % num_parts for add in range(1, num_parts)]

	list_train = list([arr_pieces[index] for index in indexes])
	train = np.vstack(list_train)
	test = arr_pieces[index_cv]

	return train, test

def get_matrix_X():
	cells = pickle.load(open('data/matrix_X.p', 'rb'))
	x = np.sum(cells, axis=0)
	cells = cells[:, x >= 1]	

	return cells

def sigmoid(x): 
  return 1.0 / (1.0 + np.exp(-x))

def get_aver_weight(aver_loss, L):
	if aver_loss is None:
		return L
	else:
		return aver_loss * 0.99 + L * 0.01

def get_labels(cells, num=2):
	kmeans = KMeans(n_clusters=num, random_state=0).fit(cells)
	return kmeans.labels_

def get_labels_2():
	cells = get_matrix_X() 
	kmeans = KMeans(n_clusters=2, random_state=1013).fit(cells)
	return kmeans.labels_

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
	return name2index_cancer[name_cancer]

def get_total_cancer():
	return 32